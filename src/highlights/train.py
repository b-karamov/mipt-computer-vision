import json
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.highlights.features.cache import FeatureCache, load_feature_cache
from src.highlights.metrics import binary_f1, spearman
from src.highlights.models.causal_tcn import CausalTCN
from src.highlights.models.losses import binary_highlight_loss, focal_dice_ranking_loss, highlight_loss


def train_from_feature_dir(
    feature_dir: str | Path,
    out_dir: str | Path,
    input_dim: int = 512,
    hidden_dim: int = 256,
    levels: int = 5,
    kernel_size: int = 3,
    dropout: float = 0.15,
    causal: bool = True,
    lookahead_steps: int = 0,
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    lambda_rank: float = 0.2,
    val_ratio: float = 0.2,
    target: str = "score",
    loss_name: str = "auto",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.5,
    dice_weight: float = 0.5,
    smoothness_weight: float = 0.0,
    entropy_weight: float = 0.0,
    feature_noise_std: float = 0.0,
    temporal_mask_prob: float = 0.0,
    split_strategy: str = "cache",
    split_seed: int = 42,
    test_ratio: float = 0.1,
    soft_score_weight: float = 0.0,
    segment_window: int = 1,
    segment_stride: int = 1,
    device: str = "auto",
) -> dict[str, object]:
    """Обучает temporal head на директории `.npz` feature-cache.

    Финальная архитектура использует `target="summary"`, `loss_name="auto"`,
    `segment_window=4`, без augmentation, soft labels и entropy penalty.
    Остальные параметры сохранены как экспериментальные переключатели для
    воспроизведения ablation-запусков из `EXPERIMENTS.md`.
    """

    feature_dir = Path(feature_dir)
    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    pred_dir = out_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    loaded_caches = [load_feature_cache(path) for path in sorted(feature_dir.glob("*.npz"))]
    caches = [cache for cache in loaded_caches if _has_training_frames(cache)]
    skipped_empty = len(loaded_caches) - len(caches)
    assigned_splits = {
        cache.video_id: _assigned_split(cache, split_strategy, split_seed, val_ratio=val_ratio, test_ratio=test_ratio)
        for cache in caches
    }
    train_items = [c for c in caches if assigned_splits[c.video_id] == "train"]
    val_items = [c for c in caches if assigned_splits[c.video_id] == "val"]
    if not train_items:
        raise ValueError(f"No train feature caches found in {feature_dir}")
    if not val_items:
        val_items = train_items

    resolved_device = _resolve_device(device)
    model = CausalTCN(
        input_dim,
        hidden_dim,
        levels,
        kernel_size,
        dropout,
        causal=causal,
        lookahead_steps=lookahead_steps,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, object] = {"epochs": [], "skipped_empty_caches": skipped_empty}
    best_val = float("inf")
    best_spearman = float("-inf")
    best_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for cache in train_items:
            x, y, _, _, _ = _cache_to_tensors(
                cache,
                resolved_device,
                target,
                soft_score_weight=soft_score_weight,
                segment_window=segment_window,
                segment_stride=segment_stride,
            )
            x = _augment_features(x, feature_noise_std=feature_noise_std, temporal_mask_prob=temporal_mask_prob)
            pred = model(x)
            if loss_name == "focal_dice_rank":
                loss = focal_dice_ranking_loss(
                    pred.squeeze(0),
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    focal_weight=focal_weight,
                    dice_weight=dice_weight,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            elif target == "summary":
                loss = binary_highlight_loss(
                    pred.squeeze(0),
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            else:
                loss = highlight_loss(
                    pred.squeeze(0),
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_spearman, val_f1, val_predictions = _evaluate(
            model,
            val_items,
            resolved_device,
            lambda_rank,
            target,
            loss_name,
            focal_alpha,
            focal_gamma,
            focal_weight,
            dice_weight,
            smoothness_weight,
            entropy_weight,
            soft_score_weight,
            segment_window,
            segment_stride,
            assigned_splits,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_spearman": val_spearman,
            "val_summary_f1_top15": val_f1,
        }
        history["epochs"].append(row)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "levels": levels,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "causal": causal,
                "lookahead_steps": lookahead_steps,
            },
            "train_config": {
                "target": target,
                "loss": loss_name,
                "lambda_rank": lambda_rank,
                "val_ratio": val_ratio,
                "focal_alpha": focal_alpha,
                "focal_gamma": focal_gamma,
                "focal_weight": focal_weight,
                "dice_weight": dice_weight,
                "smoothness_weight": smoothness_weight,
                "entropy_weight": entropy_weight,
                "feature_noise_std": feature_noise_std,
                "temporal_mask_prob": temporal_mask_prob,
                "split_strategy": split_strategy,
                "split_seed": split_seed,
                "test_ratio": test_ratio,
                "soft_score_weight": soft_score_weight,
                "segment_window": segment_window,
                "segment_stride": segment_stride,
            },
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint_payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
            (pred_dir / "val_predictions.json").write_text(
                json.dumps(val_predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if np.isfinite(val_spearman) and val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save(checkpoint_payload, checkpoint_dir / "best_spearman.pt")
            (pred_dir / "val_predictions_best_spearman.json").write_text(
                json.dumps(val_predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if np.isfinite(val_f1) and val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(checkpoint_payload, checkpoint_dir / "best_f1.pt")
            (pred_dir / "val_predictions_best_f1.json").write_text(
                json.dumps(val_predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    (metrics_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def train_from_config(config, out_dir: str | Path = "outputs") -> dict[str, object]:
    """Запускает обучение из типизированного YAML-конфига."""

    return train_from_feature_dir(
        feature_dir=config.features.cache_dir,
        out_dir=out_dir,
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        levels=config.model.levels,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        causal=config.model.causal,
        lookahead_steps=config.model.lookahead_steps,
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        lambda_rank=config.train.lambda_rank,
        val_ratio=config.train.val_ratio,
        target=config.train.target,
        loss_name=config.train.loss,
        focal_alpha=config.train.focal_alpha,
        focal_gamma=config.train.focal_gamma,
        focal_weight=config.train.focal_weight,
        dice_weight=config.train.dice_weight,
        smoothness_weight=config.train.smoothness_weight,
        entropy_weight=config.train.entropy_weight,
        feature_noise_std=config.train.feature_noise_std,
        temporal_mask_prob=config.train.temporal_mask_prob,
        split_strategy=config.train.split_strategy,
        split_seed=config.train.split_seed,
        test_ratio=config.train.test_ratio,
        soft_score_weight=config.train.soft_score_weight,
        segment_window=config.train.segment_window,
        segment_stride=config.train.segment_stride,
        device=config.device,
    )


def _evaluate(
    model,
    items: list[FeatureCache],
    device: str,
    lambda_rank: float,
    target: str = "score",
    loss_name: str = "auto",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.5,
    dice_weight: float = 0.5,
    smoothness_weight: float = 0.0,
    entropy_weight: float = 0.0,
    soft_score_weight: float = 0.0,
    segment_window: int = 1,
    segment_stride: int = 1,
    assigned_splits: dict[str, str] | None = None,
) -> tuple[float, float, float, list[dict[str, object]]]:
    """Считает validation loss, Spearman, F1@top15 и сохраняемые predictions."""

    model.eval()
    losses = []
    correlations = []
    f1s = []
    predictions = []
    with torch.no_grad():
        for cache in items:
            x, y, timestamps, gt_score, gt_summary = _cache_to_tensors(
                cache,
                device,
                target,
                soft_score_weight=soft_score_weight,
                segment_window=segment_window,
                segment_stride=segment_stride,
            )
            pred = model(x).squeeze(0)
            if loss_name == "focal_dice_rank":
                loss = focal_dice_ranking_loss(
                    pred,
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    focal_weight=focal_weight,
                    dice_weight=dice_weight,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            elif target == "summary":
                loss = binary_highlight_loss(
                    pred,
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            else:
                loss = highlight_loss(
                    pred,
                    y.squeeze(0),
                    lambda_rank=lambda_rank,
                    smoothness_weight=smoothness_weight,
                    entropy_weight=entropy_weight,
                )
            scores = torch.sigmoid(pred).detach().cpu().numpy()
            losses.append(float(loss.detach().cpu()))
            correlations.append(spearman(scores, gt_score))
            if gt_summary is not None and len(gt_summary) == len(scores):
                f1s.append(_top_ratio_f1(scores, gt_summary, ratio=0.15)["f1"])
            predictions.append(
                {
                    "video_id": cache.video_id,
                    "split": (assigned_splits or {}).get(cache.video_id, cache.split),
                    "timestamps": timestamps.astype(float).tolist(),
                    "scores": scores.astype(float).tolist(),
                    "target": gt_score.astype(float).tolist(),
                    "target_summary": gt_summary.astype(float).tolist() if gt_summary is not None else None,
                }
            )
    finite_corr = [v for v in correlations if np.isfinite(v)]
    return (
        float(np.mean(losses)),
        float(np.mean(finite_corr)) if finite_corr else float("nan"),
        float(np.mean(f1s)) if f1s else float("nan"),
        predictions,
    )


def _cache_to_tensors(
    cache: FeatureCache,
    device: str,
    target: str = "score",
    soft_score_weight: float = 0.0,
    segment_window: int = 1,
    segment_stride: int = 1,
):
    """Преобразует один `FeatureCache` в batch tensors для TCN."""

    timestamps, features, y_np, gt_score, gt_summary = _prepare_cache_arrays(
        cache,
        target=target,
        soft_score_weight=soft_score_weight,
        segment_window=segment_window,
        segment_stride=segment_stride,
    )
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)
    return x, y, timestamps, gt_score, gt_summary


def _augment_features(
    x,
    feature_noise_std: float = 0.0,
    temporal_mask_prob: float = 0.0,
):
    """Экспериментальная feature-space augmentation.

    В финальной модели выключена (`feature_noise_std=0`, `temporal_mask_prob=0`),
    потому что повышала validation F1, но ухудшала test F1.
    """

    if feature_noise_std > 0:
        x = x + feature_noise_std * x.new_empty(x.shape).normal_()
    if temporal_mask_prob > 0:
        keep = x.new_empty((*x.shape[:2], 1)).uniform_() >= temporal_mask_prob
        x = x * keep.to(dtype=x.dtype)
    return x


def _assigned_split(
    cache: FeatureCache,
    split_strategy: str = "cache",
    split_seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> str:
    """Назначает split из cache или детерминированно reshuffle-ит по `video_id`."""

    if split_strategy == "cache":
        return cache.split
    if split_strategy != "reshuffle":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")
    digest = hashlib.sha256(f"{split_seed}:{cache.video_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def _prepare_cache_arrays(
    cache: FeatureCache,
    target: str = "score",
    soft_score_weight: float = 0.0,
    segment_window: int = 1,
    segment_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Готовит features/targets и применяет segment pooling.

    Финальная модель использует `segment_window=4`: четыре соседних CLIP-вектора
    усредняются, а binary summary target берётся как max внутри окна.
    """

    target_array = _target_array(cache, target, soft_score_weight=soft_score_weight)
    gt_score = np.asarray(cache.gt_score, dtype=np.float32)
    gt_summary = np.asarray(cache.gt_summary, dtype=np.float32) if cache.gt_summary is not None else None
    timestamps = np.asarray(cache.timestamps, dtype=np.float32)
    features = np.asarray(cache.features, dtype=np.float32)
    window = max(1, int(segment_window))
    stride = max(1, int(segment_stride))
    if window <= 1:
        return timestamps, features, target_array, gt_score, gt_summary
    if len(features) < window:
        return timestamps, features, target_array, gt_score, gt_summary

    out_timestamps = []
    out_features = []
    out_target = []
    out_scores = []
    out_summary = [] if gt_summary is not None else None
    for start in range(0, len(features) - window + 1, stride):
        end = start + window
        center = start + window // 2
        out_timestamps.append(float(timestamps[min(center, len(timestamps) - 1)]))
        out_features.append(features[start:end].mean(axis=0))
        out_target.append(float(target_array[start:end].max() if target == "summary" else target_array[start:end].mean()))
        out_scores.append(float(gt_score[start:end].mean()))
        if gt_summary is not None and out_summary is not None:
            out_summary.append(float(gt_summary[start:end].max()))
    return (
        np.asarray(out_timestamps, dtype=np.float32),
        np.asarray(out_features, dtype=np.float32),
        np.asarray(out_target, dtype=np.float32),
        np.asarray(out_scores, dtype=np.float32),
        np.asarray(out_summary, dtype=np.float32) if out_summary is not None else None,
    )


def _target_array(cache: FeatureCache, target: str, soft_score_weight: float = 0.0) -> np.ndarray:
    """Выбирает training target: continuous score или binary/soft summary.

    `soft_score_weight > 0` был экспериментом с soft labels и не используется
    в финальной конфигурации.
    """

    if target == "score":
        return cache.gt_score
    if target == "summary":
        if cache.gt_summary is not None:
            summary = cache.gt_summary.astype(np.float32)
            if soft_score_weight > 0:
                score = _normalize_target_score(cache.gt_score)
                return ((1.0 - soft_score_weight) * summary + soft_score_weight * score).astype(np.float32)
            return summary
        threshold = np.percentile(cache.gt_score, 85)
        return (cache.gt_score >= threshold).astype(np.float32)
    raise ValueError(f"Unsupported training target: {target}")


def _has_training_frames(cache: FeatureCache) -> bool:
    """Проверяет, что cache содержит непустые признаки и разметку."""

    return (
        len(cache.features) > 0
        and len(cache.timestamps) > 0
        and len(cache.gt_score) > 0
        and (cache.gt_summary is None or len(cache.gt_summary) > 0)
    )


def _normalize_target_score(score: np.ndarray) -> np.ndarray:
    """Min-max нормализует continuous GT score для legacy soft-label экспериментов."""

    score = np.asarray(score, dtype=np.float32)
    if len(score) == 0:
        return score
    lo = float(score.min())
    hi = float(score.max())
    if hi - lo < 1e-8:
        return np.zeros_like(score, dtype=np.float32)
    return ((score - lo) / (hi - lo)).astype(np.float32)


def _top_ratio_f1(scores: np.ndarray, target: np.ndarray, ratio: float) -> dict[str, float]:
    """Считает F1 для top-ratio выбранных timestamp-ов."""

    if len(scores) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    k = max(1, int(round(len(scores) * ratio)))
    pred = np.zeros(len(scores), dtype=bool)
    pred[np.argsort(scores)[-k:]] = True
    return binary_f1(pred, target.astype(bool))


def _resolve_device(device: str) -> str:
    """Выбирает MPS на Apple Silicon при `device=auto`, иначе возвращает явный device."""

    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device
