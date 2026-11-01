#!/bin/sh
# Copy the MATLAB repo's datasets into the web app: photos (dataset25) and ground truth
# (dataset_sentinel_gt). Usage: scripts/fetch_data.sh [path-to-3d-reconstruction-ui-matlab]
set -e
SRC="${1:-$HOME/codes/3d-reconstruction-ui-matlab}"
DEST="$(dirname "$0")/../web/app/data"
mkdir -p "$DEST/photos" "$DEST/gt"
cp "$SRC"/dataset25/*.png "$DEST/photos/"
cp "$SRC"/dataset_sentinel_gt/*.png "$DEST/gt/"
echo "Copied $(ls "$DEST/photos" | wc -l | tr -d ' ') photos and $(ls "$DEST/gt" | wc -l | tr -d ' ') ground-truth masks to $DEST"
