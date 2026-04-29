"""
eval_metrics.py
================
cWDM 합성 DaTScan 영상의 종합 평가:
  - 영상 품질: PSNR, SSIM, RMSE
  - 임상적 유효성: Caudate/Putamen/전체 SBR, P/C Ratio (Tossici-Bolt 정의)
  - 정성적 비교: Color scale 시각화 (jet colormap)

Usage:
    python eval_metrics.py [results_dir]
    python eval_metrics.py results/brats_unet_50000

Expected directory layout:
    results_dir/
    ├── pid1/
    │   ├── sample.nii.gz   (합성 V04)
    │   └── target.nii.gz   (실제 V04)
    ├── pid2/
    │   ├── sample.nii.gz
    │   └── target.nii.gz
    ...

Output:
    - eval_metrics_{dirname}.csv       (환자별 전체 결과)
    - eval_summary_{dirname}.txt       (요약 통계)
    - figures/comparison_{pid}.png     (환자별 color scale 비교)
    - figures/sbr_scatter.png          (SBR 산점도)
    - figures/regional_sbr_scatter.png (부위별 SBR 산점도)
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# [NEW] Test set filtering — split 파일에서 PID 로딩
# ============================================================
def load_split_pids(split_file, split_name="test"):
    """
    Split JSON에서 평가 대상 PID 리스트를 로딩한다.
    make_split.py가 만든 형식과 호환:
        {"train": [...], "val": [...], "test": [...], "_meta": {...}}
    """
    import json
    with open(split_file) as f:
        data = json.load(f)
    if split_name not in data:
        raise KeyError(f"'{split_name}' not in {split_file}. "
                       f"Available: {[k for k in data if not k.startswith('_')]}")
    pids = [str(p) for p in data[split_name]]
    print(f"[load_split_pids] '{split_name}' split: {len(pids)} PIDs from {split_file}")
    return pids
    
# ============================================================
# 1. Image Quality Metrics
# ============================================================
def calc_psnr(real, synth):
    """Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = np.mean((real - synth) ** 2)
    if mse < 1e-10:
        return float("inf")
    max_val = real.max()
    return 10.0 * np.log10(max_val ** 2 / mse)


def calc_rmse(real, synth):
    """Root Mean Square Error. Lower is better."""
    return np.sqrt(np.mean((real - synth) ** 2))


def calc_ssim(real, synth):
    """Structural Similarity Index. Higher is better (max 1)."""
    data_range = max(real.max() - real.min(), synth.max() - synth.min())
    if data_range < 1e-8:
        return 0.0
    return ssim(real, synth, data_range=data_range)


def calc_mae(real, synth):
    """Mean Absolute Error. Lower is better."""
    return np.mean(np.abs(real - synth))


# ============================================================
# 2. SBR (Striatal Binding Ratio) — Tossici-Bolt et al. (2006)
# ============================================================
# ROI centers in MNI152 2mm space
ROI_CENTERS_MNI = {
    "caudate_L": (-12, 10, 8),
    "caudate_R": (12, 10, 8),
    "putamen_L": (-24, 4, 0),
    "putamen_R": (24, 4, 0),
    "occipital_L": (-24, -85, 10),
    "occipital_R": (24, -85, 10),
}
BOX_RADIUS = {
    "caudate": (5, 6, 5),
    "putamen": (6, 7, 5),
    "occipital": (8, 8, 6),
}

# PPMI DaTScan affine (MNI152 2mm)
AFFINE = np.array([
    [-2.0, 0.0, 0.0, 90.0],
    [0.0, 2.0, 0.0, -126.0],
    [0.0, 0.0, 2.0, -74.0],
    [0.0, 0.0, 0.0, 1.0],
])
INV_AFFINE = np.linalg.inv(AFFINE)
VOLUME_SHAPE = (91, 109, 91)


def mni_to_voxel(mni_xyz):
    mni = np.array([*mni_xyz, 1.0])
    vox = INV_AFFINE @ mni
    return tuple(int(round(v)) for v in vox[:3])


def make_box_mask(center_vox, radius, shape=VOLUME_SHAPE):
    mask = np.zeros(shape, dtype=bool)
    ci, cj, ck = center_vox
    ri, rj, rk = radius
    i0, i1 = max(0, ci - ri), min(shape[0], ci + ri + 1)
    j0, j1 = max(0, cj - rj), min(shape[1], cj + rj + 1)
    k0, k1 = max(0, ck - rk), min(shape[2], ck + rk + 1)
    mask[i0:i1, j0:j1, k0:k1] = True
    return mask


def build_masks(shape=VOLUME_SHAPE):
    m = {}
    print("ROI voxel coordinates (from MNI):")
    for name, mni in ROI_CENTERS_MNI.items():
        region = name.rsplit("_", 1)[0]
        vox = mni_to_voxel(mni)
        print(f"  {name:15s}: MNI{mni} → voxel{vox}")
        # 범위 확인
        for dim, v, s in zip("ijk", vox, shape):
            if v < 0 or v >= s:
                print(f"    WARNING: {dim}={v} out of bounds (0~{s-1})!")
        m[name] = make_box_mask(vox, BOX_RADIUS[region], shape)
    m["caudate"] = m["caudate_L"] | m["caudate_R"]
    m["putamen"] = m["putamen_L"] | m["putamen_R"]
    m["occipital"] = m["occipital_L"] | m["occipital_R"]
    return m


def compute_sbr(vol, masks):
    """SBR = (ROI_mean - REF_mean) / REF_mean (Tossici-Bolt et al., 2006)
    
    Tossici-Bolt 방식: caudate+putamen을 하나의 선조체 ROI로 합쳐서 계산.
    Murakami et al. (2018): "SBR = [specific binding (caudate and putamen) 
    - nonspecific binding] / nonspecific binding"
    """
    ref_mean = vol[masks["occipital"]].mean()
    if ref_mean < 1e-8:
        return {
            "caudate_sbr": np.nan,
            "putamen_sbr": np.nan,
            "mean_sbr": np.nan,
            "pc_ratio": np.nan,
        }
    cau = (vol[masks["caudate"]].mean() - ref_mean) / ref_mean
    put = (vol[masks["putamen"]].mean() - ref_mean) / ref_mean
    # 전체 SBR: caudate+putamen 합친 영역에서 계산 (Tossici-Bolt 정의)
    striatum_mean = vol[masks["caudate"] | masks["putamen"]].mean()
    mean_sbr = (striatum_mean - ref_mean) / ref_mean
    pc = put / cau if abs(cau) > 1e-6 else np.nan
    return {
        "caudate_sbr": cau,
        "putamen_sbr": put,
        "mean_sbr": mean_sbr,
        "pc_ratio": pc,
    }


# ============================================================
# 3. Visualization
# ============================================================
def plot_comparison(real, synth, pid, save_path, slice_idx=None):
    """Color scale 비교: 실제 V04 vs 합성 V04 (jet colormap)."""
    if slice_idx is None:
        slice_idx = real.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(real[:, :, slice_idx].max(), synth[:, :, slice_idx].max())

    # Real V04
    im0 = axes[0].imshow(
        real[:, :, slice_idx].T, cmap="jet", origin="lower", vmin=0, vmax=vmax
    )
    axes[0].set_title(f"Real V04 (Ground Truth)")
    axes[0].axis("off")

    # Synth V04
    im1 = axes[1].imshow(
        synth[:, :, slice_idx].T, cmap="jet", origin="lower", vmin=0, vmax=vmax
    )
    axes[1].set_title(f"Synthesized V04")
    axes[1].axis("off")

    # Difference
    diff = np.abs(real[:, :, slice_idx] - synth[:, :, slice_idx])
    im2 = axes[2].imshow(
        diff.T, cmap="hot", origin="lower", vmin=0, vmax=vmax * 0.3
    )
    axes[2].set_title(f"Absolute Difference")
    axes[2].axis("off")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Uptake")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Uptake")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="|Diff|")

    fig.suptitle(f"Patient {pid}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sbr_scatter(df, save_dir):
    """SBR 산점도: Real vs Synth."""
    # 1. 전체 SBR
    fig, ax = plt.subplots(figsize=(7, 6))
    r, p = pearsonr(df["target_mean_sbr"], df["sample_mean_sbr"])
    ax.scatter(df["target_mean_sbr"], df["sample_mean_sbr"], alpha=0.7, s=60)
    z = np.polyfit(df["target_mean_sbr"], df["sample_mean_sbr"], 1)
    x_line = np.linspace(df["target_mean_sbr"].min(), df["target_mean_sbr"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--")
    ax.set_xlabel("Real V04 SBR")
    ax.set_ylabel("Synth V04 SBR")
    ax.set_title(f"Overall SBR (r={r:.3f}, p={p:.2e})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sbr_scatter.png"), dpi=150)
    plt.close()

    # 2. 부위별 SBR
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (roi, color, label) in enumerate([
        ("caudate_sbr", "blue", "Caudate SBR"),
        ("putamen_sbr", "red", "Putamen SBR"),
        ("pc_ratio", "green", "P/C Ratio"),
    ]):
        t_col = f"target_{roi}"
        s_col = f"sample_{roi}"
        valid = df[[t_col, s_col]].dropna()
        r, p = pearsonr(valid[t_col], valid[s_col])

        axes[i].scatter(valid[t_col], valid[s_col], alpha=0.7, s=60, c=color)
        z = np.polyfit(valid[t_col], valid[s_col], 1)
        x_line = np.linspace(valid[t_col].min(), valid[t_col].max(), 100)
        axes[i].plot(x_line, np.polyval(z, x_line), "r--")
        axes[i].set_xlabel(f"Real V04 {label}")
        axes[i].set_ylabel(f"Synth V04 {label}")
        axes[i].set_title(f"{label} (r={r:.3f}, p={p:.2e})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regional_sbr_scatter.png"), dpi=150)
    plt.close()


# ============================================================
# 4. Main Evaluation
# ============================================================
def evaluate(results_dir, split_pids=None, split_name="test"):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    dirname = os.path.basename(results_dir.rstrip("/"))
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Build ROI masks
    # Detect volume shape from first patient
    first_pid = None
    for pid in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, pid)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "target.nii.gz")):
            vol = nib.load(os.path.join(d, "target.nii.gz")).get_fdata()
            vol_shape = vol.shape
            first_pid = pid
            break

    if first_pid is None:
        raise FileNotFoundError("No valid patient directories found")

    print(f"Volume shape: {vol_shape}")
    masks = build_masks(vol_shape)

    print("ROI voxel counts:")
    for k in ["caudate", "putamen", "occipital"]:
        print(f"  {k:10s}: {masks[k].sum()} voxels")
    print()

    # Evaluate each patient
    rows = []
    n_vis = 0
    max_vis = 10  # color scale 비교 이미지 최대 생성 수

    # ============================================================
    # [NEW] split_pids 필터링
    # ------------------------------------------------------------
    # results_dir에는 sampling된 환자 폴더가 있다.
    # split_pids가 주어지면 해당 PID만 평가한다.
    # ============================================================
    all_dirs = sorted([d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d))])

    if split_pids is not None:
        split_set = set(split_pids)
        pids_to_eval = [p for p in all_dirs if p in split_set]
        missing = split_set - set(all_dirs)
        if missing:
            print(f"[warning] {len(missing)} '{split_name}' PIDs declared in split "
                  f"are NOT present in {results_dir}:")
            for p in sorted(missing)[:10]:
                print(f"  - {p}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        print(f"[info] Filtered: {len(pids_to_eval)} of {len(all_dirs)} "
              f"folders match '{split_name}' split.\n")
    else:
        pids_to_eval = all_dirs
        print(f"[warning] No split filter. Evaluating ALL {len(all_dirs)} folders "
              f"(NOT a held-out evaluation).\n")
    # ============================================================

    for pid in pids_to_eval:   # [MODIFIED] was: sorted(os.listdir(results_dir))
        d = os.path.join(results_dir, pid)
        if not os.path.isdir(d):
            continue

        sample_path = os.path.join(d, "sample.nii.gz")
        target_path = os.path.join(d, "target.nii.gz")
        if not os.path.exists(sample_path) or not os.path.exists(target_path):
            continue

        try:
            synth = nib.load(sample_path).get_fdata().astype(np.float32)
            real = nib.load(target_path).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"[skip] {pid}: {e}")
            continue

        if synth.shape != real.shape:
            print(f"[skip shape] {pid}: synth={synth.shape}, real={real.shape}")
            continue

        # Image quality metrics
        psnr = calc_psnr(real, synth)
        ss = calc_ssim(real, synth)
        rmse = calc_rmse(real, synth)
        mae = calc_mae(real, synth)

        # SBR
        s_sbr = compute_sbr(synth, masks)
        t_sbr = compute_sbr(real, masks)

        row = {
            "pid": pid,
            "psnr": psnr,
            "ssim": ss,
            "rmse": rmse,
            "mae": mae,
        }
        for k, v in s_sbr.items():
            row[f"sample_{k}"] = v
        for k, v in t_sbr.items():
            row[f"target_{k}"] = v
        rows.append(row)

        # Color scale visualization (처음 max_vis명만)
        if n_vis < max_vis:
            plot_comparison(
                real, synth, pid, os.path.join(fig_dir, f"comparison_{pid}.png")
            )
            n_vis += 1

        print(
            f"{pid}: PSNR={psnr:.2f}dB SSIM={ss:.4f} RMSE={rmse:.4f} | "
            f"SBR real={t_sbr['mean_sbr']:.3f} synth={s_sbr['mean_sbr']:.3f}"
        )

    # ============================================================
    # Summary statistics
    # ============================================================
    df = pd.DataFrame(rows)
    df_clean = df.dropna()

    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY (n={len(df_clean)})")
    print(f"{'=' * 70}")

    # Image quality
    print(f"\n--- Image Quality ---")
    print(
        f"{'Metric':<10s}   {'Mean':>10s}   {'Std':>10s}   "
        f"{'Min':>10s}   {'Max':>10s}"
    )
    print("-" * 60)
    for metric in ["psnr", "ssim", "rmse", "mae"]:
        vals = df_clean[metric]
        print(
            f"{metric.upper():<10s}   {vals.mean():>10.4f}   {vals.std():>10.4f}   "
            f"{vals.min():>10.4f}   {vals.max():>10.4f}"
        )

    # SBR correlations
    print(f"\n--- SBR Correlations (Real V04 vs Synth V04) ---")
    print(
        f"{'ROI':<15s}   {'r':>8s}   {'p':>12s}   "
        f"{'Sample Mean±Std':>20s}   {'Target Mean±Std':>20s}"
    )
    print("-" * 90)
    for roi in ["caudate_sbr", "putamen_sbr", "mean_sbr", "pc_ratio"]:
        s_col = f"sample_{roi}"
        t_col = f"target_{roi}"
        valid = df_clean[[s_col, t_col]].dropna()
        if len(valid) < 3:
            continue
        r, p = pearsonr(valid[t_col], valid[s_col])
        print(
            f"{roi.upper():<15s}  {r:>8.4f}  {p:>12.2e}   "
            f"{valid[s_col].mean():>8.3f} ± {valid[s_col].std():>5.3f}   "
            f"{valid[t_col].mean():>8.3f} ± {valid[t_col].std():>5.3f}"
        )

    # Paired t-test (Real vs Synth SBR)
    print(f"\n--- Paired t-test (Real V04 SBR vs Synth V04 SBR) ---")
    for roi in ["caudate_sbr", "putamen_sbr", "mean_sbr", "pc_ratio"]:
        s_col = f"sample_{roi}"
        t_col = f"target_{roi}"
        valid = df_clean[[s_col, t_col]].dropna()
        if len(valid) < 3:
            continue
        t_stat, p_val = ttest_rel(valid[t_col], valid[s_col])
        print(f"  {roi.upper():<15s}: t={t_stat:>8.3f}, p={p_val:>10.4e}")

    # SBR scatter plots
    plot_sbr_scatter(df_clean, fig_dir)

    # [MODIFIED] split 이름을 파일명에 명시 (val/test 결과 혼동 방지)
    suffix = f"_{split_name}" if split_pids is not None else "_all"
    csv_path = f"eval_metrics_{dirname}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save summary text
    summary_path = f"eval_summary_{dirname}{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary (n={len(df_clean)})\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Image Quality:\n")
        for metric in ["psnr", "ssim", "rmse", "mae"]:
            vals = df_clean[metric]
            f.write(f"  {metric.upper()}: {vals.mean():.4f} ± {vals.std():.4f}\n")
        f.write(f"\nSBR Correlations:\n")
        for roi in ["caudate_sbr", "putamen_sbr", "mean_sbr", "pc_ratio"]:
            s_col = f"sample_{roi}"
            t_col = f"target_{roi}"
            valid = df_clean[[s_col, t_col]].dropna()
            if len(valid) < 3:
                continue
            r, p = pearsonr(valid[t_col], valid[s_col])
            f.write(f"  {roi.upper()}: r={r:.4f}, p={p:.4e}\n")
        f.write(f"\nPaired t-test (Real vs Synth SBR):\n")
        for roi in ["caudate_sbr", "putamen_sbr", "mean_sbr", "pc_ratio"]:
            s_col = f"sample_{roi}"
            t_col = f"target_{roi}"
            valid = df_clean[[s_col, t_col]].dropna()
            if len(valid) < 3:
                continue
            t_stat, p_val = ttest_rel(valid[t_col], valid[s_col])
            f.write(f"  {roi.upper()}: t={t_stat:.3f}, p={p_val:.4e}\n")
    print(f"Saved summary: {summary_path}")

    print(f"\nFigures saved to: {fig_dir}/")
    print(f"  - comparison_*.png (color scale comparisons, up to {max_vis})")
    print(f"  - sbr_scatter.png (overall SBR scatter)")
    print(f"  - regional_sbr_scatter.png (caudate/putamen/P-C ratio)")


# ============================================================
# [MODIFIED] argparse로 변경
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="cWDM 합성 결과 평가 (val/test 등 특정 split만 평가)."
    )
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--split_file", type=str, default=None,
                        help="split.json 경로. 없으면 전체 평가.")
    parser.add_argument("--split", type=str, default="test",
                        help="평가할 split 이름 ('val' 또는 'test', 기본: test)")
    args = parser.parse_args()

    pids = load_split_pids(args.split_file, args.split) if args.split_file else None
    evaluate(args.results_dir, split_pids=pids, split_name=args.split)