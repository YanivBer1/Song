#!/usr/bin/env bash
# Renders the main Reel, the 5 A/B hook variants and the cover still,
# then normalizes every MP4 to -14 LUFS / AAC 128k / 48 kHz.
#
#   npm run render:hooks                 (uses Google Fonts, needs internet)
#   REMOTION_LOCAL_FONTS=1 npm run render:hooks   (offline, public/fonts)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out

for COMP in Mouza11Attempts MouzaHookA MouzaHookB MouzaHookC MouzaHookD MouzaHookE; do
  echo "▶ rendering $COMP"
  npx remotion render "$COMP" "out/${COMP}.mp4" --codec h264 --image-format jpeg --scale 1 "$@"
done

echo "▶ cover still (frame 24)"
npx remotion still MouzaCover out/cover.png --frame=24 "$@"

bash scripts/loudnorm.sh
echo "✔ done – upload the files in out/final/"
