#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _dump_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            # Classification-only: drop detection fields if present.
            if "boxes" in it:
                it = {k: v for k, v in it.items() if k != "boxes"}
            f.write(json.dumps(it) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Stratified train/val/test split for dogs_flat.jsonl")
    p.add_argument("--src", type=Path, default=Path("data/dogs_flat.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-image", action="store_true")
    args = p.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Missing source jsonl: {args.src}")
    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1:
        raise ValueError("Require: train>0, val>=0, train+val<1")

    items: list[dict] = []
    with args.src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            if args.require_image and not Path(it["image"]).exists():
                continue
            items.append(it)

    by_class: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        by_class[int(it["class_id"])].append(it)

    rng = random.Random(args.seed)
    train_items: list[dict] = []
    val_items: list[dict] = []
    test_items: list[dict] = []

    for class_id, class_items in by_class.items():
        rng.shuffle(class_items)
        n = len(class_items)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        train_items.extend(class_items[:n_train])
        val_items.extend(class_items[n_train : n_train + n_val])
        test_items.extend(class_items[n_train + n_val :])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    _dump_jsonl(args.out_dir / "dogs_train.jsonl", train_items)
    _dump_jsonl(args.out_dir / "dogs_val.jsonl", val_items)
    _dump_jsonl(args.out_dir / "dogs_test.jsonl", test_items)

    print("train/val/test:", len(train_items), len(val_items), len(test_items))
    print("num classes:", len(by_class))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
