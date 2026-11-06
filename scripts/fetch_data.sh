#!/bin/sh
# Copy datasets into the web app (web/app/data/<dataset>/).
#   sentinel_polar: the MATLAB repo's photos (dataset25) and ground truth (dataset_sentinel_gt)
#   zebra:          real photos with no ground truth, shrunk to 320 px wide
# Usage: scripts/fetch_data.sh [path-to-3d-reconstruction-ui-matlab] [path-to-zebra-photos]
set -e
SRC="${1:-$HOME/codes/3d-reconstruction-ui-matlab}"
ZEBRA="${2:-$HOME/code/data/zebra}"
DEST="$(dirname "$0")/../web/app/data"
mkdir -p "$DEST/sentinel_polar/photos" "$DEST/sentinel_polar/gt" "$DEST/zebra/photos"
cp "$SRC"/dataset25/*.png "$DEST/sentinel_polar/photos/"
cp "$SRC"/dataset_sentinel_gt/*.png "$DEST/sentinel_polar/gt/"
echo "sentinel_polar: $(ls "$DEST/sentinel_polar/photos" | wc -l | tr -d ' ') photos, $(ls "$DEST/sentinel_polar/gt" | wc -l | tr -d ' ') ground-truth masks"
if [ -d "$ZEBRA" ]; then
  python3 - "$ZEBRA" "$DEST/zebra/photos" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1:]
for i in range(1, 37):
    Image.open("%s/%d.jpg" % (src, i)).convert("RGB").resize((320, 216), Image.LANCZOS).save("%s/%d.jpg" % (dst, i), quality=90)
PY
  echo "zebra: 36 photos"
fi
