import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


MANIFEST_FIELDS = ["video_id", "youtube_id", "local_path", "duration", "split", "download_status"]


@dataclass(frozen=True)
class MRHiSumRecord:
    """Одна строка manifest: идентификаторы, локальный mp4, split и статус загрузки."""

    video_id: str
    youtube_id: str
    local_path: Path | None
    duration: float
    split: str
    download_status: str


def write_manifest(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    """Записывает manifest CSV в стабильном формате, пригодном для повторного запуска."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def load_manifest(path: str | Path) -> list[MRHiSumRecord]:
    """Загружает manifest CSV и приводит строки к `MRHiSumRecord`."""

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [_row_to_record(row) for row in csv.DictReader(f)]


def successful_records(records: Iterable[MRHiSumRecord]) -> list[MRHiSumRecord]:
    """Оставляет только успешно скачанные видео с локальным путём."""

    return [record for record in records if record.download_status == "downloaded" and record.local_path]


def load_split_map(split_json: str | Path | None) -> dict[str, str]:
    """Читает официальный split JSON MR.HiSum, если он доступен."""

    if split_json is None:
        return {}
    path = Path(split_json)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    for split_name, key in [("train", "train_keys"), ("val", "val_keys"), ("test", "test_keys")]:
        for video_id in raw.get(key, []):
            out[str(video_id)] = split_name
    return out


def prepare_subset_manifest(
    metadata_csv: str | Path,
    out_csv: str | Path,
    videos_dir: str | Path,
    target_count: int = 500,
    split_json: str | Path | None = None,
    download: bool = True,
    max_attempts: int | None = None,
) -> list[MRHiSumRecord]:
    """Формирует subset manifest и при необходимости скачивает YouTube-видео.

    Функция используется для подготовки данных, а не во время финального inference.
    Недоступные ролики не останавливают пайплайн: они помечаются как `failed`.
    """

    split_map = load_split_map(split_json)
    rows: list[dict[str, object]] = []
    successes = 0
    attempts = 0
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if successes >= target_count:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1
            video_id = row.get("video_id") or row.get("id")
            youtube_id = row.get("youtube_id")
            if not video_id or not youtube_id:
                continue
            split = split_map.get(video_id, row.get("split", "train"))
            duration = row.get("duration", "")
            local_path = local_video_path(videos_dir, video_id, youtube_id)
            status = "downloaded" if local_path.exists() else "pending"
            if download and status != "downloaded":
                status = _download_youtube_video(youtube_id, local_path)
            if status == "downloaded":
                successes += 1
            rows.append(
                {
                    "video_id": video_id,
                    "youtube_id": youtube_id,
                    "local_path": str(local_path if status == "downloaded" else ""),
                    "duration": duration,
                    "split": split,
                    "download_status": status,
                }
            )

    write_manifest(out_csv, rows)
    return load_manifest(out_csv)


def load_gt_score(gt_path: str | Path, video_id: str) -> np.ndarray:
    """Возвращает continuous importance score для одного видео из MR.HiSum HDF5."""

    return _load_gt_array(gt_path, video_id, ("gt_score", "gtscore"))


def load_gt_summary(gt_path: str | Path, video_id: str) -> np.ndarray:
    """Возвращает бинарные GT summary labels для одного видео из MR.HiSum HDF5."""

    return _load_gt_array(gt_path, video_id, ("gt_summary", "gtsummary"))


def _load_gt_array(gt_path: str | Path, video_id: str, keys: tuple[str, ...]) -> np.ndarray:
    """Достаёт массив разметки из HDF5, поддерживая разные имена ключей."""

    with h5py.File(gt_path, "r") as h5:
        if video_id not in h5:
            raise KeyError(f"{video_id} not found in {gt_path}")
        group = h5[video_id]
        for key in keys:
            if key in group:
                return np.asarray(group[key], dtype=np.float32)
    raise KeyError(f"No {'/'.join(keys)} found for {video_id}")


def align_scores_to_timestamps(gt_score: np.ndarray, timestamps: np.ndarray, duration_sec: float | None = None) -> np.ndarray:
    """Интерполирует GT-разметку MR.HiSum на timestamp-и выбранной частоты кадров."""

    gt_score = np.asarray(gt_score, dtype=np.float32).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if len(timestamps) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(gt_score) == 0:
        return np.zeros_like(timestamps, dtype=np.float32)
    duration = float(duration_sec) if duration_sec and duration_sec > 0 else max(float(timestamps[-1] + 1.0), len(gt_score))
    gt_times = np.linspace(0.0, duration, num=len(gt_score), endpoint=False, dtype=np.float32)
    return np.interp(timestamps, gt_times, gt_score).astype(np.float32)


def _row_to_record(row: dict[str, str]) -> MRHiSumRecord:
    """Преобразует CSV-строку manifest в типизированную запись."""

    local_path = Path(row["local_path"]) if row.get("local_path") else None
    duration = float(row["duration"]) if row.get("duration") else 0.0
    return MRHiSumRecord(
        video_id=row["video_id"],
        youtube_id=row["youtube_id"],
        local_path=local_path,
        duration=duration,
        split=row["split"],
        download_status=row["download_status"],
    )


def local_video_path(videos_dir: str | Path, video_id: str, youtube_id: str) -> Path:
    """Строит безопасное имя локального mp4 для пары `video_id` и `youtube_id`."""

    safe_youtube_id = "".join(ch for ch in youtube_id if ch.isalnum() or ch in {"-", "_"})
    return Path(videos_dir) / f"{video_id}_{safe_youtube_id}.mp4"


def _download_youtube_video(youtube_id: str, out_path: Path) -> str:
    """Скачивает ролик через yt-dlp и возвращает статус для manifest.

    Используется только на этапе подготовки датасета. Финальный репозиторий
    работает с уже подготовленным manifest и локальными видео.
    """

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "mp4[height<=480]/best[height<=480]/best",
        "-o",
        str(out_path),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return "downloaded" if result.returncode == 0 and out_path.exists() else "failed"
