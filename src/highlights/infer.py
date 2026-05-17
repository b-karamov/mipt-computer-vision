import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.highlights.features.clip_encoder import CLIPFrameEncoder
from src.highlights.metrics import real_time_factor
from src.highlights.models.causal_tcn import CausalTCN
from src.highlights.postprocess import PostprocessConfig, normalize_scores, scores_to_intervals
from src.highlights.video_io import probe_video, render_preview, sample_frames


def run_inference(
    video_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    config=None,
    make_preview: bool = True,
) -> dict[str, object]:
    """Запускает полный inference и пишет demo artifacts в `out_dir`.

    Функция используется CLI и Streamlit demo. Она применяет segment pooling из
    checkpoint-а, поэтому произвольное видео обрабатывается тем же способом, что
    и validation/test cache.
    """

    started = time.perf_counter()
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)

    fps = config.features.fps if config else 1.0
    image_size = config.features.image_size if config else 224
    batch_size = config.features.batch_size if config else 32
    encoder_name = config.features.encoder if config else "ViT-B-32"
    pretrained = config.features.pretrained if config else "laion2b_s34b_b79k"
    device = config.device if config else "auto"

    sampled = list(sample_frames(video_path, fps=fps, image_size=image_size))
    timestamps = np.array([item[0] for item in sampled], dtype=np.float32)
    frames = [item[1] for item in sampled]
    encoder = CLIPFrameEncoder(encoder=encoder_name, pretrained=pretrained, device=device)
    features = encoder.encode_frames(frames, batch_size=batch_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    train_cfg = checkpoint.get("train_config", {})
    segment_window = int(train_cfg.get("segment_window", 1))
    segment_stride = int(train_cfg.get("segment_stride", 1))
    timestamps, features = _prepare_inference_arrays(timestamps, features, segment_window, segment_stride)
    resolved_device = "mps" if device == "auto" and torch.backends.mps.is_available() else ("cpu" if device == "auto" else device)
    model = CausalTCN(**model_cfg).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=resolved_device).unsqueeze(0)
        raw_scores = torch.sigmoid(model(x).squeeze(0)).detach().cpu().numpy()

    post_cfg = _postprocess_from_config(config)
    scores = normalize_scores(raw_scores, mode="minmax")
    intervals = scores_to_intervals(timestamps, scores, post_cfg, duration_sec=info.duration_sec)
    elapsed = time.perf_counter() - started
    rtf = real_time_factor(elapsed, info.duration_sec) if info.duration_sec > 0 else None

    scores_payload = {
        "video": str(video_path),
        "duration_sec": info.duration_sec,
        "processing_sec": elapsed,
        "real_time_factor": rtf,
        "timestamps": timestamps.astype(float).tolist(),
        "raw_scores": raw_scores.astype(float).tolist(),
        "scores": scores.astype(float).tolist(),
        "score_normalization": "minmax",
    }
    highlights_payload = {"video": str(video_path), "intervals": intervals}
    (out_dir / "scores.json").write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "highlights.json").write_text(json.dumps(highlights_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_path = out_dir / "timeline.png"
    plot_timeline(timestamps, scores, intervals, timeline_path)
    preview_path = None
    if make_preview:
        preview_path = render_preview(video_path, intervals, out_dir / "highlight_preview.mp4")

    return {
        "scores": scores_payload,
        "highlights": highlights_payload,
        "timeline_path": str(timeline_path),
        "preview_path": str(preview_path) if preview_path else None,
    }


def _prepare_inference_arrays(
    timestamps: np.ndarray,
    features: np.ndarray,
    segment_window: int = 1,
    segment_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Повторяет segment pooling на inference-видео согласно train config checkpoint-а."""

    window = max(1, int(segment_window))
    stride = max(1, int(segment_stride))
    if window <= 1 or len(features) < window:
        return timestamps, features
    out_timestamps = []
    out_features = []
    for start in range(0, len(features) - window + 1, stride):
        end = start + window
        center = min(start + window // 2, len(timestamps) - 1)
        out_timestamps.append(float(timestamps[center]))
        out_features.append(features[start:end].mean(axis=0))
    return np.asarray(out_timestamps, dtype=np.float32), np.asarray(out_features, dtype=np.float32)


def plot_timeline(
    timestamps: np.ndarray,
    scores: np.ndarray,
    intervals: list[dict[str, float]],
    out_path: str | Path,
    gt_scores: np.ndarray | None = None,
    gt_summary: np.ndarray | None = None,
) -> Path:
    """Рисует score timeline, prediction-интервалы и опциональные GT overlays."""

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(timestamps, scores, color="#2563eb", linewidth=1.8, label="pred score")
    if gt_scores is not None and len(gt_scores) == len(timestamps):
        gt_scores = normalize_scores(gt_scores, mode="minmax")
        ax.plot(timestamps, gt_scores, color="#16a34a", linewidth=1.6, alpha=0.85, label="GT score")
    if gt_summary is not None and len(gt_summary) == len(timestamps):
        gt_spans = _mask_to_spans(timestamps, gt_summary > 0.5)
        for idx, (start, end) in enumerate(gt_spans):
            ax.axvspan(start, end, color="#22c55e", alpha=0.18, label="GT summary" if idx == 0 else None)
    ax.set_xlabel("time, sec")
    ax.set_ylabel("highlight score")
    ax.set_ylim(0.0, 1.0)
    for interval in intervals:
        ax.axvspan(interval["start"], interval["end"], color="#f97316", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _mask_to_spans(timestamps: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Преобразует бинарную mask в интервалы для GT overlay."""

    if len(timestamps) == 0:
        return []
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    spans = []
    start = None
    for ts, active in zip(timestamps, mask):
        if active and start is None:
            start = float(ts)
        elif not active and start is not None:
            spans.append((start, float(ts)))
            start = None
    if start is not None:
        spans.append((start, float(timestamps[-1] + step)))
    return spans


def _postprocess_from_config(config) -> PostprocessConfig:
    """Достаёт postprocess-настройки из конфига или возвращает финальные defaults."""

    if config is None:
        return PostprocessConfig(0.35, 0.65, 0.45, 2.0, 1.5, 0.15)
    return PostprocessConfig(
        smooth_alpha=config.postprocess.smooth_alpha,
        start_threshold=config.postprocess.start_threshold,
        end_threshold=config.postprocess.end_threshold,
        min_duration_sec=config.postprocess.min_duration_sec,
        merge_gap_sec=config.postprocess.merge_gap_sec,
        max_summary_ratio=config.postprocess.max_summary_ratio,
    )
