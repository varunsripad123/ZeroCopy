from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from zerocopy.models.videomae_encoder import VideoMAEEncoder, read_video_frames_rgb


class _FakeProcessor:
    def __init__(self) -> None:
        self.last_inputs: list[np.ndarray] | None = None

    def __call__(self, frames, return_tensors="pt"):
        self.last_inputs = list(frames)
        tensor = torch.ones(1, len(self.last_inputs), 3, 2, 2)
        return {"pixel_values": tensor}


class _FakeConfig:
    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames


class _FakeModel(torch.nn.Module):
    def __init__(self, num_frames: int) -> None:
        super().__init__()
        self.config = _FakeConfig(num_frames)
        self._device = torch.device("cpu")

    def eval(self):
        return self

    def to(self, device):
        self._device = torch.device(device)
        return self

    def forward(self, *, pixel_values):
        assert pixel_values.shape[1] == self.config.num_frames
        hidden = torch.zeros(1, self.config.num_frames + 1, 4)
        hidden[:, 0, :] = torch.tensor([3.0, 0.0, 4.0, 0.0])
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.mark.parametrize("frame_count,required", [(4, 8), (10, 4)])
def test_encode_respects_frame_count(frame_count, required):
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frame_count)]
    processor = _FakeProcessor()
    model = _FakeModel(required)

    encoder = VideoMAEEncoder(model_name="stub")
    encoder.processor = processor
    encoder.model = model
    encoder._device = torch.device("cpu")  # type: ignore[attr-defined]

    embedding = encoder.encode(frames)

    assert processor.last_inputs is not None
    assert len(processor.last_inputs) == required
    assert embedding.shape == (4,)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_read_video_frames_rgb(tmp_path):
    video_path = tmp_path / "sample.mp4"
    height, width = 2, 3
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height)
    )
    assert writer.isOpened()

    originals: list[np.ndarray] = []
    for idx in range(3):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = idx * 30  # B
        frame[..., 1] = idx * 40  # G
        frame[..., 2] = idx * 50  # R
        writer.write(frame)
        originals.append(frame)
    writer.release()

    frames = read_video_frames_rgb(video_path, num_frames=2)
    assert len(frames) == 2
    assert frames[0].shape == (height, width, 3)
    # Verify BGR to RGB conversion
    assert frames[0][0, 0, 0] == originals[0][0, 0, 2]
    assert frames[0][0, 0, 2] == originals[0][0, 0, 0]


def test_read_video_frames_rgb_missing(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        read_video_frames_rgb(missing)


def test_stub_encoder_is_deterministic():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    encoder = VideoMAEEncoder(use_stub=True)
    encoder.load()

    embedding_a = encoder.encode(frames)
    embedding_b = encoder.encode(frames)
    assert embedding_a.shape == (encoder.embedding_dim,)
    np.testing.assert_allclose(embedding_a, embedding_b)

    altered = frames.copy()
    altered[0] = np.ones((2, 2, 3), dtype=np.uint8)
    embedding_c = encoder.encode(altered)
    assert not np.allclose(embedding_a, embedding_c)
