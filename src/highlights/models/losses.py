import torch
from torch.nn import functional as F


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1, max_pairs: int = 2048) -> torch.Tensor:
    """Штрафует пары, где более важный timestamp получил меньший score."""

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    n = pred.numel()
    if n < 2:
        return pred.new_tensor(0.0)
    diff_target = target[:, None] - target[None, :]
    mask = diff_target > margin
    pairs = mask.nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return pred.new_tensor(0.0)
    if len(pairs) > max_pairs:
        pairs = pairs[torch.randperm(len(pairs), device=pairs.device)[:max_pairs]]
    i = pairs[:, 0]
    j = pairs[:, 1]
    return F.relu(margin - (pred[i] - pred[j])).mean()


def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Регуляризует резкие скачки соседних sigmoid scores.

    В финальной модели используется с весом `0.05`.
    """

    scores = torch.sigmoid(pred).reshape(-1)
    if scores.numel() < 2:
        return scores.new_tensor(0.0)
    return torch.abs(scores[1:] - scores[:-1]).mean()


def entropy_penalty_loss(pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Экспериментальный entropy penalty для более уверенных score-ов.

    В финальной архитектуре не используется (`entropy_weight=0.0`), потому что
    ablation не дал прироста качества.
    """

    prob = torch.sigmoid(pred).reshape(-1)
    entropy = -(prob * torch.log(prob.clamp_min(eps)) + (1.0 - prob) * torch.log((1.0 - prob).clamp_min(eps)))
    return 1.0 - entropy.mean()


def highlight_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_rank: float = 0.2,
    smoothness_weight: float = 0.0,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """Legacy regression loss для continuous GT score.

    Не используется в финальной модели: после анализа ошибок мы перешли на
    бинарные summary labels и `binary_highlight_loss`.
    """

    huber = F.huber_loss(torch.sigmoid(pred), target)
    rank = pairwise_ranking_loss(torch.sigmoid(pred), target)
    smoothness = temporal_smoothness_loss(pred)
    entropy = entropy_penalty_loss(pred)
    return huber + lambda_rank * rank + smoothness_weight * smoothness + entropy_weight * entropy


def binary_highlight_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_rank: float = 0.2,
    smoothness_weight: float = 0.0,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """Финальный loss: weighted BCE + pairwise ranking + optional regularizers."""

    target = target.reshape_as(pred).float()
    positives = target.sum()
    negatives = target.numel() - positives
    if positives > 0:
        pos_weight = (negatives / positives).clamp(min=1.0, max=20.0)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(pred, target)
    rank = pairwise_ranking_loss(torch.sigmoid(pred), target, margin=0.05)
    smoothness = temporal_smoothness_loss(pred)
    entropy = entropy_penalty_loss(pred)
    return bce + lambda_rank * rank + smoothness_weight * smoothness + entropy_weight * entropy


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Экспериментальный focal loss для редких положительных highlight labels."""

    target = target.reshape_as(pred).float()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = torch.sigmoid(pred)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Экспериментальный Dice loss для overlap с бинарной summary mask."""

    target = target.reshape_as(pred).float()
    prob = torch.sigmoid(pred).reshape(-1)
    target = target.reshape(-1)
    intersection = (prob * target).sum()
    denominator = prob.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def focal_dice_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_rank: float = 0.5,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.5,
    dice_weight: float = 0.5,
    smoothness_weight: float = 0.0,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """Экспериментальная комбинация focal + dice + ranking.

    Сохранена для воспроизведения таблицы экспериментов, но финальная модель
    использует `binary_highlight_loss`.
    """

    target = target.reshape_as(pred).float()
    focal = sigmoid_focal_loss(pred, target, alpha=focal_alpha, gamma=focal_gamma)
    dice = soft_dice_loss(pred, target)
    rank = pairwise_ranking_loss(torch.sigmoid(pred), target, margin=0.05)
    smoothness = temporal_smoothness_loss(pred)
    entropy = entropy_penalty_loss(pred)
    return (
        focal_weight * focal
        + dice_weight * dice
        + lambda_rank * rank
        + smoothness_weight * smoothness
        + entropy_weight * entropy
    )
