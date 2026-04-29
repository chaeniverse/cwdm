#!/bin/bash
set -e

SRC=/workspace/DaTSCAN-pair
DST=/workspace/DaTSCAN-nii

for visit in SC V04; do
    mkdir -p "$DST/$visit"
    ls "$SRC/$visit" | xargs -P 8 -I{} bash -c '
        src="'"$SRC/$visit"'/{}"
        out="'"$DST/$visit"'"
        if [ ! -f "$out/{}.nii.gz" ]; then
            dcm2niix -z y -o "$out" -f "{}" "$src" > /dev/null 2>&1
        fi
    '
done

echo "SC converted: $(ls $DST/SC/*.nii.gz 2>/dev/null | wc -l)"
echo "V04 converted: $(ls $DST/V04/*.nii.gz 2>/dev/null | wc -l)"
