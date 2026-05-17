from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureCache:
    """Cached признаки одного видео после CLIP extraction."""

    video_id: str
    split: str
    timestamps: np.ndarray
    features: np.ndarray
    gt_score: np.ndarray
    gt_summary: np.ndarray | None = None


def save_feature_cache(path: str | Path, cache: FeatureCache) -> None:
    """Сохраняет CLIP embeddings, timestamp-и и GT в сжатый `.npz` файл."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": np.array(cache.video_id),
        "split": np.array(cache.split),
        "timestamps": np.asarray(cache.timestamps, dtype=np.float32),
        "features": np.asarray(cache.features, dtype=np.float32),
        "gt_score": np.asarray(cache.gt_score, dtype=np.float32),
    }
    if cache.gt_summary is not None:
        payload["gt_summary"] = np.asarray(cache.gt_summary, dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_feature_cache(path: str | Path) -> FeatureCache:
    """Загружает `.npz` cache и возвращает типизированный `FeatureCache`."""

    with np.load(Path(path), allow_pickle=False) as data:
        gt_summary = np.asarray(data["gt_summary"], dtype=np.float32) if "gt_summary" in data.files else None
        return FeatureCache(
            video_id=str(data["video_id"].item()),
            split=str(data["split"].item()),
            timestamps=np.asarray(data["timestamps"], dtype=np.float32),
            features=np.asarray(data["features"], dtype=np.float32),
            gt_score=np.asarray(data["gt_score"], dtype=np.float32),
            gt_summary=gt_summary,
        )
