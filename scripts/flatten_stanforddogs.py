#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/dogs_flat.jsonl"))
    p.add_argument(
        "--require-image",
        action="store_true",
        help="Kept for compatibility; classification mode always uses existing images.",
    )
    args = p.parse_args()

    img_root = args.data_root / "images" / "Images"

    if not img_root.exists():
        raise FileNotFoundError(f"Missing images root: {img_root}")

    breed_names = sorted([d.name for d in img_root.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(breed_names)}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_missing = 0

    with args.out.open("w", encoding="utf-8") as f:
        for class_name in breed_names:
            breed_img_dir = img_root / class_name
            if not breed_img_dir.exists():
                continue

            for img_path in sorted(breed_img_dir.glob("*.jpg")):
                if args.require_image and not img_path.exists():
                    n_missing += 1
                    continue

                item = {
                    "image": str(img_path.as_posix()),
                    "class_name": class_name,
                    "class_id": class_to_idx[class_name],
                }
                f.write(json.dumps(item) + "\n")
                n_written += 1

    print(f"wrote {n_written} lines to {args.out}")
    if n_missing:
        print(f"skipped {n_missing} entries with missing images")
    print(f"num classes: {len(class_to_idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
