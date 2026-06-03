#!/usr/bin/env python3
"""
download_coco_vww.py — Build a local VWW-style test set from COCO 2017.

Downloads:
  1. COCO 2017 val annotations    (~241 MB, JSON)  — always
  2. COCO 2017 val images          (~778 MB, JPEG)  — always
  3. COCO 2017 train images        (~18  GB, JPEG)  — only with --use-train

Organises output as:
  vww_data/
    person/       images that contain at least one person
    non_person/   images with no person

Then run the evaluation with:
  python eval_vww.py --dataset-dir vww_data --no-bootstrap

Usage:
  python download_coco_vww.py                          # val only  (~5 000 images)
  python download_coco_vww.py --max-images 500         # balanced subset (250+250)
  python download_coco_vww.py --use-train              # val + train (~123 000 images)
  python download_coco_vww.py --use-train --max-images 10000  # 5 000+5 000
  python download_coco_vww.py --out-dir ./vww_data
"""

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_IMAGES_URL   = "http://images.cocodataset.org/zips/val2017.zip"
TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
PERSON_CLASS_ID  = 1   # COCO category id for "person"


def download_file(url: str, dest: Path, desc: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip]  {dest.name} already downloaded")
        return
    print(f"[down]  {desc}  →  {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path, marker: str):
    """Extract zip only if `marker` subdirectory/file doesn't already exist."""
    if (out_dir / marker).exists():
        print(f"[skip]  {zip_path.name} already extracted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name}  →  {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _split_ids(annotations_json: Path):
    """Return (person_ids, no_person_ids, id_to_filename) from a COCO instances JSON."""
    with open(annotations_json) as f:
        coco = json.load(f)
    person_image_ids = {ann["image_id"] for ann in coco["annotations"]
                        if ann["category_id"] == PERSON_CLASS_ID}
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    person_ids  = [i for i in id_to_name if i     in person_image_ids]
    no_per_ids  = [i for i in id_to_name if i not in person_image_ids]
    return person_ids, no_per_ids, id_to_name


def build_vww(splits: list[tuple[Path, Path]], out_dir: Path, max_images: int | None):
    """
    splits : list of (annotations_json, images_dir) pairs (val first, then train)
    Copies images into out_dir/person/ and out_dir/non_person/.
    When max_images is set the quota is split equally across splits, then balanced.
    """
    all_person, all_noperson = [], []   # list of (src_path,) per class

    for ann_json, images_dir in splits:
        split_name = ann_json.stem  # e.g. instances_val2017
        print(f"[build] Reading {split_name} …")
        person_ids, no_per_ids, id_to_name = _split_ids(ann_json)
        print(f"        raw: person={len(person_ids)}  no_person={len(no_per_ids)}")

        if max_images:
            # Distribute quota proportionally across splits (called once per split here,
            # so each split gets max_images // len(splits) // 2 per class).
            half = (max_images // len(splits)) // 2
            rng  = random.Random(42)
            person_ids = rng.sample(person_ids, min(half, len(person_ids)))
            no_per_ids = rng.sample(no_per_ids, min(half, len(no_per_ids)))

        for img_id in person_ids:
            p = images_dir / id_to_name[img_id]
            if p.exists():
                all_person.append(p)
        for img_id in no_per_ids:
            p = images_dir / id_to_name[img_id]
            if p.exists():
                all_noperson.append(p)

    print(f"[build] Total available — person={len(all_person)}  no_person={len(all_noperson)}")

    for srcs, folder in [(all_person, "person"), (all_noperson, "non_person")]:
        dest = out_dir / folder
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in srcs:
            shutil.copy2(src, dest / src.name)
            copied += 1
        print(f"[build] {folder}: {copied} images copied → {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",    type=Path, default=Path("vww_data"))
    ap.add_argument("--cache-dir",  type=Path, default=Path("coco_cache"))
    ap.add_argument("--max-images", type=int,  default=None,
                    help="Total images to select (balanced person/no_person). "
                         "Default: all available")
    ap.add_argument("--use-train",  action="store_true",
                    help="Also download COCO 2017 train images (~18 GB) to "
                         "supplement val2017 and reach larger dataset sizes")
    ap.add_argument("--skip-images", action="store_true",
                    help="Skip downloading images (use if already cached)")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Annotations (val + train in one zip) ──────────────────────────────────
    ann_zip = args.cache_dir / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, ann_zip, "COCO 2017 annotations (~241 MB)")
    ann_dir = args.cache_dir / "annotations_extracted"
    extract_zip(ann_zip, ann_dir, marker="annotations")
    ann_val   = ann_dir / "annotations" / "instances_val2017.json"
    ann_train = ann_dir / "annotations" / "instances_train2017.json"

    # ── Val images ────────────────────────────────────────────────────────────
    if not args.skip_images:
        val_zip = args.cache_dir / "val2017.zip"
        download_file(VAL_IMAGES_URL, val_zip, "COCO 2017 val images (~778 MB)")
        val_img_dir = args.cache_dir / "val_images_extracted"
        extract_zip(val_zip, val_img_dir, marker="val2017")
    else:
        val_img_dir = args.cache_dir / "images_extracted"
    val_images = val_img_dir / "val2017"

    splits = [(ann_val, val_images)]

    # ── Train images (optional) ───────────────────────────────────────────────
    if args.use_train:
        print("[info]  --use-train: downloading COCO 2017 train images (~18 GB)")
        if not args.skip_images:
            train_zip = args.cache_dir / "train2017.zip"
            download_file(TRAIN_IMAGES_URL, train_zip, "COCO 2017 train images (~18 GB)")
            train_img_dir = args.cache_dir / "train_images_extracted"
            extract_zip(train_zip, train_img_dir, marker="train2017")
        else:
            train_img_dir = args.cache_dir / "train_images_extracted"
        train_images = train_img_dir / "train2017"
        splits.append((ann_train, train_images))

    # ── Build VWW layout ──────────────────────────────────────────────────────
    build_vww(splits, args.out_dir, args.max_images)
    print(f"\nDone.  Run evaluation with:\n"
          f"  python eval_vww.py --dataset-dir {args.out_dir} --no-bootstrap")


if __name__ == "__main__":
    main()
