from pathlib import Path

import numpy as np

from src.highlights.datasets.mrhisum import (
    align_scores_to_timestamps,
    load_gt_score,
    load_gt_summary,
    load_manifest,
    successful_records,
)
from src.highlights.features.cache import FeatureCache, load_feature_cache, save_feature_cache
from src.highlights.features.clip_encoder import CLIPFrameEncoder
from src.highlights.video_io import probe_video, sample_frames


def extract_features_from_config(config) -> int:
    """Строит `.npz` cache для всех успешно скачанных видео из manifest.

    Финальная конфигурация использует CLIP ViT-B/32 при 1 fps. Ранее проверенный
    режим 2 fps не входит в финальную архитектуру, потому что ухудшил validation
    F1 и почти удваивал стоимость feature extraction.
    """

    records = successful_records(load_manifest(config.dataset.manifest))
    encoder = CLIPFrameEncoder(config.features.encoder, config.features.pretrained, config.device)
    count = 0
    for record in records:
        if record.local_path is None:
            continue
        out_path = config.features.cache_dir / f"{record.video_id}.npz"
        if out_path.exists():
            _ensure_summary_in_cache(out_path, config.dataset.gt)
            count += 1
            continue
        info = probe_video(record.local_path)
        sampled = list(sample_frames(record.local_path, fps=config.features.fps, image_size=config.features.image_size))
        timestamps = np.array([item[0] for item in sampled], dtype=np.float32)
        frames = [item[1] for item in sampled]
        features = encoder.encode_frames(frames, batch_size=config.features.batch_size)
        gt_score = load_gt_score(config.dataset.gt, record.video_id)
        gt_summary = load_gt_summary(config.dataset.gt, record.video_id)
        aligned = align_scores_to_timestamps(gt_score, timestamps, duration_sec=info.duration_sec)
        aligned_summary = align_scores_to_timestamps(gt_summary, timestamps, duration_sec=info.duration_sec)
        save_feature_cache(
            out_path,
            FeatureCache(
                video_id=record.video_id,
                split=record.split,
                timestamps=timestamps,
                features=features,
                gt_score=aligned,
                gt_summary=(aligned_summary >= 0.5).astype(np.float32),
            ),
        )
        count += 1
    return count


def _ensure_summary_in_cache(path: Path, gt_path: Path) -> None:
    """Обновляет старый cache, если в нём ещё нет бинарных GT summary labels."""

    cache = load_feature_cache(path)
    if cache.gt_summary is not None and len(cache.gt_summary) == len(cache.timestamps):
        return
    gt_summary = load_gt_summary(gt_path, cache.video_id)
    duration = float(cache.timestamps[-1] + 1.0) if len(cache.timestamps) else float(len(gt_summary))
    aligned_summary = align_scores_to_timestamps(gt_summary, cache.timestamps, duration_sec=duration)
    save_feature_cache(
        path,
        FeatureCache(
            video_id=cache.video_id,
            split=cache.split,
            timestamps=cache.timestamps,
            features=cache.features,
            gt_score=cache.gt_score,
            gt_summary=(aligned_summary >= 0.5).astype(np.float32),
        ),
    )
