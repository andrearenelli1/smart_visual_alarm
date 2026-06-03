#!/usr/bin/env bash
# Download COCO 2017 images + annotations into ./coco_data/
# Resume-safe: wget -c continues partial downloads.
#
# Sizes:
#   train2017.zip   ~18 GB
#   val2017.zip     ~ 1 GB
#   annotations     ~241 MB
#
# Usage:
#   bash download_coco.sh              # full download
#   bash download_coco.sh --val-only   # only val2017 (~1 GB, for quick test)

set -euo pipefail

VAL_ONLY=false
[[ "${1:-}" == "--val-only" ]] && VAL_ONLY=true

ROOT="$(cd "$(dirname "$0")" && pwd)/coco_data"
mkdir -p "$ROOT"/{train2017,val2017,annotations}

BASE="http://images.cocodataset.org"

download() {
    local url="$1"
    local dest="$2"
    echo "[download] $url → $dest"
    wget -c --show-progress -O "$dest" "$url"
}

extract() {
    local zip="$1"
    local dir="$2"
    echo "[extract]  $zip"
    unzip -q -n "$zip" -d "$dir"
    rm -f "$zip"
}

# ── Annotations (always) ───────────────────────────────────────────────────
ANN_ZIP="$ROOT/annotations_trainval2017.zip"
if [ ! -f "$ROOT/annotations/instances_val2017.json" ]; then
    download "$BASE/annotations/annotations_trainval2017.zip" "$ANN_ZIP"
    extract  "$ANN_ZIP" "$ROOT"
else
    echo "[skip]     annotations already present"
fi

# ── Val images ────────────────────────────────────────────────────────────
VAL_COUNT=$(find "$ROOT/val2017" -name "*.jpg" 2>/dev/null | wc -l)
if [ "$VAL_COUNT" -lt 5000 ]; then
    VAL_ZIP="$ROOT/val2017.zip"
    download "$BASE/zips/val2017.zip" "$VAL_ZIP"
    extract  "$VAL_ZIP" "$ROOT"
else
    echo "[skip]     val2017 already present ($VAL_COUNT images)"
fi

# ── Train images ──────────────────────────────────────────────────────────
if [ "$VAL_ONLY" = true ]; then
    echo "[skip]     --val-only: skipping train2017"
else
    TRAIN_COUNT=$(find "$ROOT/train2017" -name "*.jpg" 2>/dev/null | wc -l)
    if [ "$TRAIN_COUNT" -lt 118000 ]; then
        TRAIN_ZIP="$ROOT/train2017.zip"
        download "$BASE/zips/train2017.zip" "$TRAIN_ZIP"
        extract  "$TRAIN_ZIP" "$ROOT"
    else
        echo "[skip]     train2017 already present ($TRAIN_COUNT images)"
    fi
fi

echo ""
echo "Done. COCO data in: $ROOT"
echo "  val2017  images : $(find "$ROOT/val2017"  -name '*.jpg' | wc -l)"
echo "  train2017 images: $(find "$ROOT/train2017" -name '*.jpg' | wc -l)"
