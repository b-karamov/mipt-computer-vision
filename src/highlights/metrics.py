import numpy as np


def binary_f1(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Считает precision, recall и F1 для бинарных масок highlight/не-highlight."""

    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def spearman(pred_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """Считает Spearman rank correlation без зависимости от scipy."""

    pred = np.asarray(pred_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if len(pred) != len(target):
        raise ValueError("pred_scores and target_scores must have the same length")
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(target == target[0]):
        return float("nan")
    pred_rank = _rankdata(pred)
    target_rank = _rankdata(target)
    corr = np.corrcoef(pred_rank, target_rank)[0, 1]
    return round(float(corr), 10)


def real_time_factor(processing_seconds: float, video_duration_seconds: float) -> float:
    """Возвращает долю времени обработки от длительности видео."""

    if video_duration_seconds <= 0:
        raise ValueError("video_duration_seconds must be positive")
    return processing_seconds / video_duration_seconds


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Присваивает средние ранги с корректной обработкой одинаковых значений."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks
