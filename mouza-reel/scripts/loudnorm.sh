#!/usr/bin/env bash
# Two-pass EBU R128 loudness normalization to -14 LUFS (Instagram target),
# true peak -1.5 dBTP, AAC 128 kbps, 48 kHz. Video stream is copied untouched.
# Uses the ffmpeg that ships with Remotion (`npx remotion ffmpeg`), so no
# extra install is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out/final

FF="npx --no-install remotion ffmpeg"

for IN in out/*.mp4; do
  NAME=$(basename "$IN")
  echo "▶ loudnorm $NAME"
  STATS=$($FF -hide_banner -nostats -i "$IN" -vn -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
  get() { echo "$STATS" | grep "\"$1\"" | sed -E 's/.*: "([^"]+)".*/\1/'; }
  $FF -hide_banner -loglevel error -y -i "$IN" -c:v copy \
    -af "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=$(get input_i):measured_TP=$(get input_tp):measured_LRA=$(get input_lra):measured_thresh=$(get input_thresh):offset=$(get target_offset):linear=true" \
    -ar 48000 -c:a aac -b:a 128k -movflags +faststart "out/final/$NAME"
done
echo "✔ normalized files in out/final/"
