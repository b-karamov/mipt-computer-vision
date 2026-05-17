import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """1D convolution с левым padding для causal inference.

    `lookahead_steps > 0` использовался только в экспериментах с ограниченным
    заглядыванием в будущее; финальная модель использует `lookahead_steps=0`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        lookahead_steps: int = 0,
    ) -> None:
        """Создаёт causal convolution с заданной дилатацией и lookahead."""

        super().__init__()
        total_padding = (kernel_size - 1) * dilation
        self.right_padding = min(max(0, int(lookahead_steps)), total_padding)
        self.left_padding = total_padding - self.right_padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет causal convolution к тензору `(B,C,T)`."""

        x = F.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)


class SamePadConv1d(nn.Module):
    """Экспериментальная non-causal convolution с симметричным padding.

    В финальную near-real-time архитектуру не входит: она требует будущие кадры.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        """Создаёт симметричный convolution для bidirectional-экспериментов."""

        super().__init__()
        total_padding = (kernel_size - 1) * dilation
        self.left_padding = total_padding // 2
        self.right_padding = total_padding - self.left_padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет same-padding convolution к тензору `(B,C,T)`."""

        x = F.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)


class CausalBlock(nn.Module):
    """Residual TCN-блок из двух temporal convolution слоёв."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool = True,
        lookahead_steps: int = 0,
    ) -> None:
        """Собирает residual-блок из двух temporal convolution слоёв."""

        super().__init__()
        first = (
            CausalConv1d(channels, channels, kernel_size, dilation, lookahead_steps)
            if causal
            else SamePadConv1d(channels, channels, kernel_size, dilation)
        )
        second = (
            CausalConv1d(channels, channels, kernel_size, dilation, lookahead_steps)
            if causal
            else SamePadConv1d(channels, channels, kernel_size, dilation)
        )
        self.net = nn.Sequential(
            first,
            nn.ReLU(),
            nn.Dropout(dropout),
            second,
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает residual-сумму входа и temporal-преобразования."""

        return x + self.net(x)


class CausalTCN(nn.Module):
    """Лёгкий temporal head поверх CLIP embeddings.

    Финальная модель: `input_dim=512`, `hidden_dim=64`, `levels=2`,
    `kernel_size=3`, `dropout=0.4`, `causal=True`, `lookahead_steps=0`.
    Параметры `causal=False`, `lookahead_steps>0` и более глубокие варианты
    сохранены только для воспроизведения архитектурных ablation-экспериментов.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        levels: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.15,
        causal: bool = True,
        lookahead_steps: int = 0,
    ) -> None:
        """Создаёт TCN head с входным projection, residual blocks и 1D head."""

        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                CausalBlock(hidden_dim, kernel_size, dilation=2**idx, dropout=dropout, causal=causal)
                if lookahead_steps == 0
                else CausalBlock(
                    hidden_dim,
                    kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                    causal=causal,
                    lookahead_steps=lookahead_steps,
                )
                for idx in range(levels)
            ]
        )
        self.head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Принимает `(batch, time, feature)` и возвращает logits `(batch, time)`."""

        if x.ndim != 3:
            raise ValueError("CausalTCN expects input shape (batch, time, feature)")
        x = x.transpose(1, 2)
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)
