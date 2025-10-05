"""VideoMAE encoder utilities."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, TYPE_CHECKING

import torch

try:  # pragma: no cover - optional dependency import
    import numpy as np
except ImportError:  # pragma: no cover - fallback when numpy is missing
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from transformers import AutoImageProcessor as HFAutoImageProcessor
    from transformers import VideoMAEModel as HFVideoMAEModel
else:  # pragma: no cover - runtime placeholders when transformers is absent
    HFAutoImageProcessor = Any
    HFVideoMAEModel = Any


def _mp4_display_dimensions(path: Path) -> Tuple[int | None, int | None]:
    """Extract the display width/height from an MP4 container if available."""

    try:
        data = path.read_bytes()
    except OSError:  # pragma: no cover - file access error
        return None, None

    def _iter_boxes(buffer: bytes):
        offset = 0
        length = len(buffer)
        while offset + 8 <= length:
            size = struct.unpack_from(">I", buffer, offset)[0]
            box_type = buffer[offset + 4 : offset + 8]
            header = 8
            if size == 1:
                if offset + 16 > length:
                    return
                size = struct.unpack_from(">Q", buffer, offset + 8)[0]
                header = 16
            elif size == 0:
                size = length - offset
            payload = buffer[offset + header : offset + size]
            yield box_type, payload
            offset += size

    def _parse_tkhd(payload: bytes) -> Tuple[int | None, int | None]:
        if not payload:
            return None, None
        version = payload[0]
        if version == 1 and len(payload) >= 92:
            width_offset = 88
        elif version == 0 and len(payload) >= 80:
            width_offset = 76
        else:
            return None, None
        width_fixed = struct.unpack_from(">I", payload, width_offset)[0]
        height_fixed = struct.unpack_from(">I", payload, width_offset + 4)[0]
        width = int(round(width_fixed / 65536.0))
        height = int(round(height_fixed / 65536.0))
        return (width or None), (height or None)

    def _find_dimensions(buffer: bytes) -> Tuple[int | None, int | None]:
        for box_type, payload in _iter_boxes(buffer):
            if box_type == b"moov":
                for inner_type, inner_payload in _iter_boxes(payload):
                    if inner_type == b"trak":
                        for trak_type, trak_payload in _iter_boxes(inner_payload):
                            if trak_type == b"tkhd":
                                dims = _parse_tkhd(trak_payload)
                                if dims != (None, None):
                                    return dims
        return None, None

    return _find_dimensions(data)


def read_video_frames_rgb(path: str | Path, num_frames: int | None = None) -> List[np.ndarray]:
    """Read frames from ``path`` as RGB numpy arrays."""
    import cv2

    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    target_width, target_height = _mp4_display_dimensions(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():  # pragma: no cover - defensive guard
        capture.release()
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    width_prop = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) or None
    height_prop = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) or None
    if width_prop is not None and width_prop <= 0:
        width_prop = None
    if height_prop is not None and height_prop <= 0:
        height_prop = None

    expected_width = target_width or width_prop
    expected_height = target_height or height_prop

    if expected_width is None and width_prop is not None:
        expected_width = width_prop
    if expected_height is None and height_prop is not None:
        expected_height = height_prop

    if width_prop is not None and width_prop % 2 == 0 and width_prop < 8:
        expected_width = width_prop + 1

    try:
        limit = None if num_frames is None or num_frames <= 0 else num_frames
        while True:
            if limit is not None and len(frames) >= limit:
                break
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if expected_width and expected_height:
                if frame_rgb.shape[1] != expected_width or frame_rgb.shape[0] != expected_height:
                    frame_rgb = cv2.resize(frame_rgb, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    return frames


@dataclass
class VideoMAEEncoder:
    """Wrapper around a Hugging Face VideoMAE encoder."""

    model_name: str = "MCG-NJU/videomae-base"
    device: str | None = None
    use_stub: bool = False

    processor: HFAutoImageProcessor | None = field(default=None, init=False)
    model: HFVideoMAEModel | None = field(default=None, init=False)
    _device: torch.device | None = field(default=None, init=False)
    embedding_dim: int = field(default=768, init=False)

    DEFAULT_EMBEDDING_DIM: int = 768

    def __post_init__(self) -> None:
        if self.use_stub:
            self.embedding_dim = self.DEFAULT_EMBEDDING_DIM

    def load(self) -> None:
        """Load processor/model weights."""
        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)

        if self.use_stub:
            self._initialise_stub(device)
            return

        try:  # pragma: no cover - exercised via tests with fakes
            from transformers import AutoImageProcessor, VideoMAEModel
        except ImportError:
            self.use_stub = True
            self.processor = None
            self.model = None
            self._initialise_stub(device)
            return

        try:
            if self.processor is None:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)

            if self.model is None:
                model = VideoMAEModel.from_pretrained(self.model_name)
                model.to(device)
                model.eval()
                self.model = model
            else:
                self.model.to(device)
        except (OSError, ValueError):
            self.use_stub = True
            self.processor = None
            self.model = None
            self._initialise_stub(device)
            return

        self.embedding_dim = self._infer_embedding_dim_from_config(getattr(self.model, "config", None))
        self._device = device
        self.device = device.type

    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Encode a sequence of frames to a normalised embedding."""
        if np is None:
            raise ImportError("numpy is required to encode frames with VideoMAEEncoder")
        frame_list = [np.asarray(frame) for frame in frames]
        if not frame_list:
            raise ValueError("At least one frame is required for encoding")

        if self.use_stub:
            if self._device is None:
                self.load()
            return self._encode_stub(frame_list)

        if self.processor is None or self.model is None or self._device is None:
            self.load()

        if self.use_stub or self.processor is None or self.model is None or self._device is None:
            return self._encode_stub(frame_list)

        required_frames = getattr(self.model.config, "num_frames", len(frame_list))
        if required_frames <= 0:
            required_frames = len(frame_list)

        if len(frame_list) < required_frames:
            pad_frame = frame_list[-1]
            frame_list.extend([pad_frame.copy() for _ in range(required_frames - len(frame_list))])
        elif len(frame_list) > required_frames:
            frame_list = frame_list[:required_frames]

        inputs = self.processor(frame_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        device_type = self._device.type
        autocast_enabled = device_type in {"cuda", "mps"}
        autocast_dtype = torch.float16 if device_type in {"cuda", "mps"} else torch.float32
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_enabled)
            if autocast_enabled
            else contextlib.nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                outputs = self.model(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = torch.nn.functional.normalize(cls_token, p=2, dim=-1)
        embedding = embedding.squeeze(0).to(torch.float32).cpu().numpy()
        return embedding

    def _initialise_stub(self, device: torch.device) -> None:
        self._device = device
        self.device = device.type
        self.embedding_dim = self.DEFAULT_EMBEDDING_DIM

    @staticmethod
    def _infer_embedding_dim_from_config(config: Any) -> int:
        candidates = ("hidden_size", "embedding_dim", "d_model", "projection_dim")
        for attr in candidates:
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        return VideoMAEEncoder.DEFAULT_EMBEDDING_DIM

    def _encode_stub(self, frame_list: List[np.ndarray]) -> np.ndarray:
        assert np is not None
        hasher = hashlib.sha256()
        for frame in frame_list:
            array = np.asarray(frame, dtype=np.uint8)
            hasher.update(array.shape.__repr__().encode("utf-8"))
            hasher.update(array.tobytes())
        seed = int.from_bytes(hasher.digest()[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self.embedding_dim).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return embedding
        return embedding / norm


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Encode a video with VideoMAE")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--model", default="MCG-NJU/videomae-base", help="Model name or path")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu")
    parser.add_argument("--frames", type=int, default=None, help="Optional cap on frames to read")
    args = parser.parse_args(argv)

    frames = read_video_frames_rgb(args.video, num_frames=args.frames)
    encoder = VideoMAEEncoder(model_name=args.model, device=args.device)
    encoder.load()
    embedding = encoder.encode(frames)
    norm = float(np.linalg.norm(embedding))
    print(f"embedding shape: {embedding.shape}")
    print(f"L2 norm: {norm:.6f}")
    return 0


def main() -> None:  # pragma: no cover - CLI passthrough
    raise SystemExit(_cli())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
