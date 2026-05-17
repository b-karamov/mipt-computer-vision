from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Пути к MR.HiSum metadata, GT-разметке, локальным видео и manifest."""

    name: str
    metadata: Path
    gt: Path
    videos_dir: Path
    manifest: Path


@dataclass(frozen=True)
class FeatureConfig:
    """Параметры CLIP-экстракции; в финале используется ViT-B/32 при 1 fps."""

    cache_dir: Path
    encoder: str
    pretrained: str
    fps: float
    image_size: int
    batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    """Параметры temporal head; финальная архитектура — causal TCN hidden=64, levels=2."""

    input_dim: int
    hidden_dim: int
    levels: int
    kernel_size: int
    dropout: float
    causal: bool = True
    lookahead_steps: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Параметры обучения и экспериментальные переключатели.

    Финальная конфигурация использует `target=summary`, `loss=auto`,
    `segment_window=4`, `smoothness_weight=0.05`, `soft_score_weight=0.0`,
    `feature_noise_std=0.0`, `temporal_mask_prob=0.0`, `entropy_weight=0.0`.
    Поля `focal_*`, `dice_weight`, `soft_score_weight`, `lookahead`-совместимые
    настройки и augmentation-параметры оставлены только для воспроизведения
    ablation-экспериментов из EXPERIMENTS.md.
    """

    epochs: int
    lr: float
    weight_decay: float
    lambda_rank: float
    val_ratio: float
    patience: int
    target: str = "score"
    loss: str = "auto"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    focal_weight: float = 0.5
    dice_weight: float = 0.5
    smoothness_weight: float = 0.0
    entropy_weight: float = 0.0
    feature_noise_std: float = 0.0
    temporal_mask_prob: float = 0.0
    split_strategy: str = "cache"
    split_seed: int = 42
    test_ratio: float = 0.1
    soft_score_weight: float = 0.0
    segment_window: int = 1
    segment_stride: int = 1


@dataclass(frozen=True)
class PostprocessConfig:
    """Параметры сглаживания, hysteresis и ограничения длины итогового summary."""

    smooth_alpha: float
    start_threshold: float
    end_threshold: float
    min_duration_sec: float
    merge_gap_sec: float
    max_summary_ratio: float


@dataclass(frozen=True)
class PipelineConfig:
    """Полная конфигурация запуска: данные, features, модель, train и postprocess."""

    seed: int
    device: str
    dataset: DatasetConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    postprocess: PostprocessConfig
    repo_root: Path


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    """Преобразует относительный путь из YAML в путь относительно корня репозитория."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _require(mapping: dict[str, Any], key: str) -> Any:
    """Достаёт обязательное поле конфига и падает с понятной ошибкой при пропуске."""

    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path, repo_root: str | Path | None = None) -> PipelineConfig:
    """Читает YAML-конфиг и возвращает типизированный `PipelineConfig`.

    Экспериментальные поля поддерживаются ради воспроизводимости ablation-таблиц,
    но финальный запуск использует только значения из `configs/clip_tcn_mrhisum.yaml`.
    """

    config_path = Path(path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    dataset_raw = _require(raw, "dataset")
    features_raw = _require(raw, "features")
    model_raw = _require(raw, "model")
    train_raw = _require(raw, "train")
    post_raw = _require(raw, "postprocess")

    return PipelineConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        repo_root=root,
        dataset=DatasetConfig(
            name=str(_require(dataset_raw, "name")),
            metadata=_resolve_path(_require(dataset_raw, "metadata"), root),
            gt=_resolve_path(_require(dataset_raw, "gt"), root),
            videos_dir=_resolve_path(_require(dataset_raw, "videos_dir"), root),
            manifest=_resolve_path(_require(dataset_raw, "manifest"), root),
        ),
        features=FeatureConfig(
            cache_dir=_resolve_path(_require(features_raw, "cache_dir"), root),
            encoder=str(_require(features_raw, "encoder")),
            pretrained=str(_require(features_raw, "pretrained")),
            fps=float(_require(features_raw, "fps")),
            image_size=int(_require(features_raw, "image_size")),
            batch_size=int(_require(features_raw, "batch_size")),
        ),
        model=ModelConfig(
            input_dim=int(_require(model_raw, "input_dim")),
            hidden_dim=int(_require(model_raw, "hidden_dim")),
            levels=int(_require(model_raw, "levels")),
            kernel_size=int(_require(model_raw, "kernel_size")),
            dropout=float(_require(model_raw, "dropout")),
            causal=bool(model_raw.get("causal", True)),
            lookahead_steps=int(model_raw.get("lookahead_steps", 0)),
        ),
        train=TrainConfig(
            epochs=int(_require(train_raw, "epochs")),
            lr=float(_require(train_raw, "lr")),
            weight_decay=float(_require(train_raw, "weight_decay")),
            lambda_rank=float(_require(train_raw, "lambda_rank")),
            val_ratio=float(_require(train_raw, "val_ratio")),
            patience=int(_require(train_raw, "patience")),
            target=str(train_raw.get("target", "score")),
            loss=str(train_raw.get("loss", "auto")),
            focal_alpha=float(train_raw.get("focal_alpha", 0.75)),
            focal_gamma=float(train_raw.get("focal_gamma", 2.0)),
            focal_weight=float(train_raw.get("focal_weight", 0.5)),
            dice_weight=float(train_raw.get("dice_weight", 0.5)),
            smoothness_weight=float(train_raw.get("smoothness_weight", 0.0)),
            entropy_weight=float(train_raw.get("entropy_weight", 0.0)),
            feature_noise_std=float(train_raw.get("feature_noise_std", 0.0)),
            temporal_mask_prob=float(train_raw.get("temporal_mask_prob", 0.0)),
            split_strategy=str(train_raw.get("split_strategy", "cache")),
            split_seed=int(train_raw.get("split_seed", raw.get("seed", 42))),
            test_ratio=float(train_raw.get("test_ratio", 0.1)),
            soft_score_weight=float(train_raw.get("soft_score_weight", 0.0)),
            segment_window=int(train_raw.get("segment_window", 1)),
            segment_stride=int(train_raw.get("segment_stride", 1)),
        ),
        postprocess=PostprocessConfig(
            smooth_alpha=float(_require(post_raw, "smooth_alpha")),
            start_threshold=float(_require(post_raw, "start_threshold")),
            end_threshold=float(_require(post_raw, "end_threshold")),
            min_duration_sec=float(_require(post_raw, "min_duration_sec")),
            merge_gap_sec=float(_require(post_raw, "merge_gap_sec")),
            max_summary_ratio=float(_require(post_raw, "max_summary_ratio")),
        ),
    )
