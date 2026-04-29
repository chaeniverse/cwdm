# PPMI DaTScan Data Preparation

PPMI raw DICOM → cWDM-ready NIfTI 변환 파이프라인입니다.

## Pipeline 순서

1. `make_pair.py` — PD 그룹에서 SC와 V04 모두 가진 subject 추출 (symlink)
2. `convert_nii.py` — DICOM → NIfTI 변환 (dcm2niix 기반, 병렬 처리)
3. `scan_bad_volumes.py` — Degenerate volume 검출 (QC)
4. `../scripts/make_split.py` — PID-level train/val/test 7:1.5:1.5 split

## 사용법

각 스크립트는 입출력 경로가 하드코딩되어 있으니 환경에 맞게 수정 후 실행하세요.

### 1. Pair 추출
```bash
python make_pair.py
```
SC/V04 paired PD subject만 추출하여 `/workspace/DaTSCAN-pair/{SC,V04}/<pid>_<imageID>` 형식으로 symlink 생성.

### 2. DICOM → NIfTI 변환
```bash
python convert_nii.py
```
8 CPU 병렬로 `dcm2niix` 실행. `convert_nii.sh`는 bash 버전 대안.

### 3. QC (Quality Control)
```bash
python scan_bad_volumes.py
```
SBR 계산 시 문제될 degenerate / NaN volume을 출력.

### 4. Train/Val/Test Split
```bash
cd ..
python scripts/make_split.py
```
출력: `split.json` (seed=42, ratios=[0.7, 0.15, 0.15])

## Output

- `split.json` — 본 연구에 사용된 train/val/test split
  - n=748 paired subjects (train 523, val 112, test 113)
  - seed=42