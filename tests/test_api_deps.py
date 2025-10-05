from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _reload_modules():
    import zerocopy.config as config_module
    import zerocopy.models.videomae_encoder as encoder_module
    import zerocopy.api.deps as deps_module

    importlib.reload(config_module)
    importlib.reload(encoder_module)
    importlib.reload(deps_module)
    deps_module.reset_dependencies()
    return config_module, encoder_module, deps_module


def test_get_faiss_store_uses_stub_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROCOPY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROCOPY_VIDEOMAE_ENABLED", "false")

    _, encoder_module, deps_module = _reload_modules()

    store = deps_module.get_faiss_store()
    assert store.dim == encoder_module.VideoMAEEncoder.DEFAULT_EMBEDDING_DIM

    encoder = deps_module.get_videomae_encoder()
    assert encoder.use_stub is True
    assert encoder.embedding_dim == encoder_module.VideoMAEEncoder.DEFAULT_EMBEDDING_DIM

    deps_module.reset_dependencies()


def test_get_faiss_store_real_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROCOPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ZEROCOPY_VIDEOMAE_ENABLED", raising=False)

    class FakeAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str):
            return cls()

        def __call__(self, frames, return_tensors="pt"):
            frame_list = list(frames)
            tensor = torch.ones(1, len(frame_list), 3, 2, 2)
            return {"pixel_values": tensor}

    class FakeVideoMAEModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=12, num_frames=4)
            self._device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_name: str):
            return cls()

        def to(self, device):
            self._device = torch.device(device)
            return self

        def eval(self):
            return self

        def forward(self, *, pixel_values):
            batch, _, _, _height, _width = pixel_values.shape
            hidden = torch.zeros(batch, self.config.num_frames + 1, self.config.hidden_size)
            hidden[:, 0, :] = torch.arange(1, self.config.hidden_size + 1, dtype=torch.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.VideoMAEModel = FakeVideoMAEModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _, encoder_module, deps_module = _reload_modules()

    store = deps_module.get_faiss_store()
    assert store.dim == FakeVideoMAEModel().config.hidden_size

    encoder = deps_module.get_videomae_encoder()
    assert encoder.use_stub is False
    assert encoder.embedding_dim == FakeVideoMAEModel().config.hidden_size

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    embedding = encoder.encode(frames)
    assert embedding.shape == (FakeVideoMAEModel().config.hidden_size,)

    deps_module.reset_dependencies()
