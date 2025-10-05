"""VideoMAE encoder utilities."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, TYPE_CHECKING

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


def read_video_frames_rgb(path: str | Path, num_frames: int | None = None) -> List[np.ndarray]:
    """Read frames from ``path`` as RGB numpy arrays."""
    import cv2

    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():  # pragma: no cover - defensive guard
        capture.release()
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        limit = None if num_frames is None or num_frames <= 0 else num_frames
        while True:
            if limit is not None and len(frames) >= limit:
                break
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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

    processor: HFAutoImageProcessor | None = field(default=None, init=False)
    model: HFVideoMAEModel | None = field(default=None, init=False)
    _device: torch.device | None = field(default=None, init=False)

    def load(self) -> None:
        """Load processor/model weights."""
        from transformers import AutoImageProcessor, VideoMAEModel

        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)

        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        if self.model is None:
            model = VideoMAEModel.from_pretrained(self.model_name)
            model.to(device)
            model.eval()
            self.model = model
        else:
            self.model.to(device)

        self._device = device
        self.device = device.type

    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Encode a sequence of frames to a normalised embedding."""
        if np is None:
            raise ImportError("numpy is required to encode frames with VideoMAEEncoder")
        if self.processor is None or self.model is None or self._device is None:
            self.load()
        assert self.processor is not None and self.model is not None and self._device is not None

        frame_list = [np.asarray(frame) for frame in frames]
        if not frame_list:
            raise ValueError("At least one frame is required for encoding")

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
