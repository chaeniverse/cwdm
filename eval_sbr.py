"""
DaTScan SBR evaluation in MNI152 2mm space.

Computes Striatal Binding Ratios for synthesized (sample) vs ground-truth
(target) V04 volumes, then reports Pearson correlations across patients.

ROI definitions are axis-aligned box ROIs centered at standard MNI coordinates
for caudate, putamen, and occipital cortex (reference). This is a reasonable
proxy for PPMI's standard SBR pipeline without needing an external atlas.

Usage:
    python eval_sbr.py
      (expects results under ./results/brats_unet_50000/<pid>/sample.nii.gz
                                                          target.nii.gz)
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------- ROI config (MNI 152 2mm) ----------------------------------------
ROI_CENTERS_MNI = {
    'caudate_L':     (-12,  10,   8),
    'caudate_R':     ( 12,  10,   8),
    'putamen_L':     (-24,   4,   0),
    'putamen_R':     ( 24,   4,   0),
    'occipital_L':   (-24, -85,  10),   # reference region
    'occipital_R':   ( 24, -85,  10),
}
BOX_RADIUS = {
    'caudate':   (5, 6, 5),   # voxel radii (2mm voxels)
    'putamen':   (6, 7, 5),
    'occipital': (8, 8, 6),
}

# Affine for PPMI DaTScan volumes in MNI152 2mm space.
AFFINE = np.array([
    [-2.0,   0.0,   0.0,   90.0],
    [ 0.0,   2.0,   0.0, -108.0],
    [ 0.0,   0.0,   2.0,  -90.0],
    [ 0.0,   0.0,   0.0,    1.0],
])
INV_AFFINE = np.linalg.inv(AFFINE)
VOLUME_SHAPE = (91, 109, 91)

# Default results directory — adjust if your ITERATIONS differ.
DEFAULT_RESULTS_DIR = 'results/brats_unet_50000'


# ---------- helpers ---------------------------------------------------------
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


def build_masks():
    m = {}
    for name, mni in ROI_CENTERS_MNI.items():
        region = name.rsplit('_', 1)[0]
        m[name] = make_box_mask(mni_to_voxel(mni), BOX_RADIUS[region])
    m['caudate']   = m['caudate_L']   | m['caudate_R']
    m['putamen']   = m['putamen_L']   | m['putamen_R']
    m['occipital'] = m['occipital_L'] | m['occipital_R']
    return m


def compute_sbr(vol, masks):
    """SBR = (ROI_mean - REF_mean) / REF_mean for each ROI."""
    ref_mean = vol[masks['occipital']].mean()
    if ref_mean < 1e-8:
        return {'caudate_sbr': np.nan, 'putamen_sbr': np.nan,
                'mean_sbr': np.nan, 'pc_ratio': np.nan}
    cau = (vol[masks['caudate']].mean() - ref_mean) / ref_mean
    put = (vol[masks['putamen']].mean() - ref_mean) / ref_mean
    pc  = put / cau if abs(cau) > 1e-6 else np.nan
    return {
        'caudate_sbr': cau,
        'putamen_sbr': put,
        'mean_sbr':    (cau + put) / 2,
        'pc_ratio':    pc,
    }


# ---------- main ------------------------------------------------------------
def evaluate(results_dir):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f'results_dir not found: {results_dir}')

    masks = build_masks()
    print('ROI voxel counts:')
    for k in ['caudate', 'putamen', 'occipital']:
        print(f'  {k:10s}: {masks[k].sum()} voxels')
    print()

    rows = []
    for pid in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, pid)
        if not os.path.isdir(d):
            continue
        try:
            s = nib.load(os.path.join(d, 'sample.nii.gz')).get_fdata()
            t = nib.load(os.path.join(d, 'target.nii.gz')).get_fdata()
        except FileNotFoundError:
            continue
        s_sbr = compute_sbr(s, masks)
        t_sbr = compute_sbr(t, masks)
        row = {'pid': pid}
        for k, v in s_sbr.items():
            row[f'sample_{k}'] = v
        for k, v in t_sbr.items():
            row[f'target_{k}'] = v
        rows.append(row)

    df = pd.DataFrame(rows).dropna()
    print(f'Evaluated {len(df)} patients\n')

    print(f"{'ROI':<10s}   {'r':>8s}   {'p':>10s}     "
          f"{'sample mean±std':>22s}   {'target mean±std':>22s}")
    print('-' * 90)
    for roi in ['caudate_sbr', 'putamen_sbr', 'mean_sbr', 'pc_ratio']:
        s_col = f'sample_{roi}'
        t_col = f'target_{roi}'
        r, p = pearsonr(df[s_col], df[t_col])
        print(f'{roi.upper():<10s}  {r:>8.4f}  {p:>10.2e}     '
              f'{df[s_col].mean():>8.3f} ± {df[s_col].std():>5.3f}     '
              f'{df[t_col].mean():>8.3f} ± {df[t_col].std():>5.3f}')

    out_csv = f'sbr_results_{os.path.basename(results_dir.rstrip("/"))}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved to: {out_csv}')


if __name__ == '__main__':
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    evaluate(target_dir)
