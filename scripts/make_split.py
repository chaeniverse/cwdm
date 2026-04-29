"""
scripts/make_split.py
=====================
PPMI DaTScan paired 데이터를 train/val/test로 환자(subject) 단위 분할.
재현성을 위해 seed를 고정하고 결과를 JSON으로 저장한다.

Usage:
    python scripts/make_split.py /workspace/DaTSCAN-nii \
        --output split.json --seed 42 --ratios 0.7 0.15 0.15
"""
import os
import sys
import json
import argparse
import random


def list_paired_pids(data_dir, source_visit="SC", target_visit="V04"):
    """SC와 V04에 모두 존재하는 PID만 반환 (datscanloader와 같은 로직)."""
    src_dir = os.path.join(data_dir, source_visit)
    tgt_dir = os.path.join(data_dir, target_visit)
    src_pids = {f.split("_")[0] for f in os.listdir(src_dir) if f.endswith(".nii.gz")}
    tgt_pids = {f.split("_")[0] for f in os.listdir(tgt_dir) if f.endswith(".nii.gz")}
    return sorted(src_pids & tgt_pids)


def split_pids(pids, ratios=(0.7, 0.15, 0.15), seed=42):
    """PID 리스트를 환자 단위로 train/val/test 분할."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = random.Random(seed)
    pids = list(pids)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # 잔여를 모두 test로 → 반올림 오차 안전 처리
    return {
        "train": sorted(pids[:n_train]),
        "val":   sorted(pids[n_train:n_train + n_val]),
        "test":  sorted(pids[n_train + n_val:]),
        "_meta": {
            "seed": seed,
            "ratios": list(ratios),
            "total": n,
            "source_visit": "SC",
            "target_visit": "V04",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="DaTSCAN-nii 폴더 (SC/, V04/ 하위 폴더 포함)")
    parser.add_argument("--output", type=str, default="split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float,
                        default=[0.7, 0.15, 0.15],
                        help="train val test 비율")
    args = parser.parse_args()

    pids = list_paired_pids(args.data_dir)
    print(f"[make_split] paired PIDs found: {len(pids)}")

    split = split_pids(pids, ratios=tuple(args.ratios), seed=args.seed)
    print(f"  train: {len(split['train'])}")
    print(f"  val:   {len(split['val'])}")
    print(f"  test:  {len(split['test'])}")

    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[make_split] saved → {args.output}")


if __name__ == "__main__":
    main()