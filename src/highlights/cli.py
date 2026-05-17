import argparse
from pathlib import Path

from src.highlights.config import load_config
from src.highlights.datasets.mrhisum import prepare_subset_manifest
from src.highlights.extract import extract_features_from_config
from src.highlights.infer import run_inference
from src.highlights.train import train_from_config


def main(argv: list[str] | None = None) -> None:
    """Точка входа `python -m src.highlights.cli`."""

    parser = argparse.ArgumentParser(prog="highlights", description="CLIP + MR.HiSum highlight detection")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-mrhisum")
    prep.add_argument("--metadata", default="data/raw/mrhisum_metadata.csv")
    prep.add_argument("--split-json", default=None)
    prep.add_argument("--videos-dir", default="data/videos/mrhisum")
    prep.add_argument("--target-count", type=int, default=500)
    prep.add_argument("--max-attempts", type=int, default=None)
    prep.add_argument("--out", default="data/manifests/mrhisum_subset.csv")
    prep.add_argument("--no-download", action="store_true")

    extract = sub.add_parser("extract-features")
    extract.add_argument("--config", default="configs/clip_tcn_mrhisum.yaml")

    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/clip_tcn_mrhisum.yaml")
    train.add_argument("--out-dir", default="outputs")

    infer = sub.add_parser("infer")
    infer.add_argument("--config", default="configs/clip_tcn_mrhisum.yaml")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--video", required=True)
    infer.add_argument("--out-dir", default="outputs/demo")
    infer.add_argument("--no-preview", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare-mrhisum":
        records = prepare_subset_manifest(
            metadata_csv=args.metadata,
            out_csv=args.out,
            videos_dir=args.videos_dir,
            target_count=args.target_count,
            split_json=args.split_json,
            download=not args.no_download,
            max_attempts=args.max_attempts,
        )
        ok = sum(1 for record in records if record.download_status == "downloaded")
        print(f"Wrote {args.out}; downloaded={ok}; rows={len(records)}")
    elif args.command == "extract-features":
        cfg = load_config(args.config, repo_root=Path.cwd())
        count = extract_features_from_config(cfg)
        print(f"Feature caches ready: {count}")
    elif args.command == "train":
        cfg = load_config(args.config, repo_root=Path.cwd())
        history = train_from_config(cfg, out_dir=args.out_dir)
        print(f"Training epochs: {len(history['epochs'])}")
    elif args.command == "infer":
        cfg = load_config(args.config, repo_root=Path.cwd())
        result = run_inference(args.video, args.checkpoint, args.out_dir, config=cfg, make_preview=not args.no_preview)
        print(f"Wrote {result['timeline_path']}")
        if result["preview_path"]:
            print(f"Wrote {result['preview_path']}")


if __name__ == "__main__":
    main()
