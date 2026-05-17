from dataclasses import dataclass
from typing import Iterable

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class CLIPFrameEncoder:
    """Кодирует RGB-кадры в L2-normalized CLIP embeddings.

    В финальной архитектуре используется `ViT-B-32` с pretrained=`openai`.
    Параметры модели не обучаются; тренируется только temporal head поверх cache.
    """

    encoder: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"

    def __post_init__(self) -> None:
        """Загружает CLIP, выбирает MPS/CPU и замораживает веса encoder-а."""

        self.torch = torch
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(self.encoder, pretrained=self.pretrained)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode_frames(self, frames: Iterable[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Преобразует последовательность кадров `(H,W,3)` в матрицу `(T,512)`."""

        batches: list[np.ndarray] = []
        current = []
        for frame in frames:
            current.append(self.preprocess(Image.fromarray(frame)))
            if len(current) >= batch_size:
                batches.append(self._encode_batch(current))
                current = []
        if current:
            batches.append(self._encode_batch(current))
        if not batches:
            return np.empty((0, 512), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def _encode_batch(self, tensors: list[object]) -> np.ndarray:
        """Кодирует один batch PIL-preprocessed tensor-ов без градиентов."""

        torch = self.torch
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return features.detach().cpu().float().numpy()
