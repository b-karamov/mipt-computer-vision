from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Базовые свойства видео, нужные для timestamp alignment и RTF."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def probe_video(path: str | Path) -> VideoInfo:
    """Открывает видео через OpenCV и возвращает fps, размер и длительность."""

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height, duration_sec=duration)


def sample_frames(path: str | Path, fps: float = 1.0, image_size: int = 224) -> Iterator[tuple[float, np.ndarray]]:
    """Сэмплирует RGB-кадры с заданной частотой для CLIP encoder-а."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0
    stride = max(1, int(round(source_fps / fps)))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            timestamp = frame_idx / source_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if image_size:
                rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            yield timestamp, rgb
        frame_idx += 1
    cap.release()


def render_preview(video_path: str | Path, intervals: list[dict[str, float]], out_path: str | Path) -> Path | None:
    """Склеивает найденные highlight-интервалы в короткий mp4 preview."""

    video_path = Path(video_path)
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened() or not intervals:
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ranges = [(int(i["start"] * fps), int(i["end"] * fps)) for i in intervals]
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if any(start <= frame_idx < end for start, end in ranges):
            writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return out_path if out_path.exists() else None
