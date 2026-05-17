from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostprocessConfig:
    """Пороговые параметры EMA smoothing, hysteresis и summary budget."""

    smooth_alpha: float
    start_threshold: float
    end_threshold: float
    min_duration_sec: float
    merge_gap_sec: float
    max_summary_ratio: float


def smooth_scores(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Сглаживает score timeline экспоненциальным скользящим средним."""

    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    if alpha >= 1.0:
        return scores.copy()
    smoothed = np.empty_like(scores, dtype=np.float32)
    smoothed[0] = scores[0]
    for i in range(1, len(scores)):
        smoothed[i] = alpha * scores[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def normalize_scores(scores: np.ndarray, mode: str = "none") -> np.ndarray:
    """Нормализует scores; финальный inference использует min-max для timeline."""

    scores = np.asarray(scores, dtype=np.float32)
    if mode in {"none", "", None}:
        return scores.copy()
    if mode != "minmax":
        raise ValueError(f"Unsupported score normalization mode: {mode}")
    if len(scores) == 0:
        return scores.copy()
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def scores_to_intervals(
    timestamps: np.ndarray,
    scores: np.ndarray,
    cfg: PostprocessConfig,
    duration_sec: float | None = None,
) -> list[dict[str, float]]:
    """Преобразует per-frame scores в список интервалов highlight preview.

    Основной путь — hysteresis thresholds. Если пики слишком слабые и интервалы
    не нашлись, используется fallback top-budget на `max_summary_ratio`.
    """

    timestamps = np.asarray(timestamps, dtype=np.float32)
    scores = smooth_scores(np.asarray(scores, dtype=np.float32), cfg.smooth_alpha)
    if len(timestamps) == 0 or len(scores) == 0:
        return []
    if len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must have the same length")

    step = _infer_step(timestamps)
    duration = float(duration_sec) if duration_sec is not None else float(timestamps[-1] + step)
    raw = _hysteresis_intervals(timestamps, scores, cfg, step, duration)
    if not raw:
        raw = _top_budget_intervals(timestamps, scores, cfg, step, duration)
    merged = _merge_intervals(raw, cfg.merge_gap_sec, scores, timestamps, step)
    budget = max(step, duration * cfg.max_summary_ratio)
    if _interval_duration(merged) > budget * 1.25:
        budgeted = _top_budget_intervals(timestamps, scores, cfg, step, duration)
        merged = _merge_intervals(budgeted, cfg.merge_gap_sec, scores, timestamps, step)
    return merged


def _infer_step(timestamps: np.ndarray) -> float:
    """Оценивает шаг timestamp-ов по медиане положительных разностей."""

    if len(timestamps) < 2:
        return 1.0
    diffs = np.diff(timestamps)
    positive = diffs[diffs > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def _hysteresis_intervals(
    timestamps: np.ndarray,
    scores: np.ndarray,
    cfg: PostprocessConfig,
    step: float,
    duration: float,
) -> list[dict[str, float]]:
    """Строит интервалы по двум порогам: start и end."""

    intervals: list[dict[str, float]] = []
    active = False
    start_idx = 0
    for idx, score in enumerate(scores):
        if not active and score >= cfg.start_threshold:
            active = True
            start_idx = idx
        elif active and score < cfg.end_threshold:
            intervals.extend(_make_interval(timestamps, scores, start_idx, idx, step, duration, cfg.min_duration_sec))
            active = False
    if active:
        intervals.extend(_make_interval(timestamps, scores, start_idx, len(scores) - 1, step, duration, cfg.min_duration_sec))
    return intervals


def _make_interval(
    timestamps: np.ndarray,
    scores: np.ndarray,
    start_idx: int,
    end_idx_exclusive: int,
    step: float,
    duration: float,
    min_duration: float,
) -> list[dict[str, float]]:
    """Создаёт один интервал и отбрасывает слишком короткие кандидаты."""

    start = float(timestamps[start_idx])
    end = min(float(timestamps[end_idx_exclusive]) if end_idx_exclusive < len(timestamps) else duration, duration)
    if end <= start:
        end = min(start + step, duration)
    if end - start < min_duration:
        return []
    span_scores = scores[start_idx : max(start_idx + 1, end_idx_exclusive)]
    return [{"start": round(start, 3), "end": round(end, 3), "score": round(float(span_scores.mean()), 4)}]


def _top_budget_intervals(
    timestamps: np.ndarray,
    scores: np.ndarray,
    cfg: PostprocessConfig,
    step: float,
    duration: float,
) -> list[dict[str, float]]:
    """Fallback: выбирает top-scored timestamps в рамках бюджета summary."""

    budget = max(step, duration * cfg.max_summary_ratio)
    selected_count = min(len(scores), max(1, int(round(budget / step))))
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(scores)[-selected_count:]] = True
    intervals = []
    start_idx = None
    for idx, active in enumerate(selected):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            intervals.extend(_make_interval(timestamps, scores, start_idx, idx, step, duration, min_duration=0.0))
            start_idx = None
    if start_idx is not None:
        intervals.extend(_make_interval(timestamps, scores, start_idx, len(scores), step, duration, min_duration=0.0))
    return intervals


def _interval_duration(intervals: list[dict[str, float]]) -> float:
    """Считает суммарную длительность списка интервалов."""

    return float(sum(max(0.0, interval["end"] - interval["start"]) for interval in intervals))


def _merge_intervals(
    intervals: list[dict[str, float]],
    merge_gap: float,
    scores: np.ndarray,
    timestamps: np.ndarray,
    step: float,
) -> list[dict[str, float]]:
    """Склеивает соседние интервалы, если разрыв между ними меньше `merge_gap`."""

    if not intervals:
        return []
    merged = [dict(intervals[0])]
    for interval in intervals[1:]:
        last = merged[-1]
        if interval["start"] - last["end"] <= merge_gap:
            last_duration = max(last["end"] - last["start"], step)
            interval_duration = max(interval["end"] - interval["start"], step)
            last["end"] = interval["end"]
            weighted = (last["score"] * last_duration + interval["score"] * interval_duration) / (
                last_duration + interval_duration
            )
            last["score"] = round(float(weighted), 4)
        else:
            merged.append(dict(interval))
    return merged
