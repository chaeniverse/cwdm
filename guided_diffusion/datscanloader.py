"""
DaTScan SC -> V04 pair dataset loader.

=========================================================
WHY THIS FILE EXISTS (compared to original bratsloader.py):
=========================================================
- BraTS is 4 modalities per patient (t1n/t1c/t2w/t2f). We have only 2 visits
  (SC as input, V04 as target). So the whole "which of 4 to generate?" logic
  is gone; it's now a simple source->target pair.
- BraTS filenames are "BraTS-GLI-00000-000-t1n.nii.gz"; our filenames are
  "{Subject}_{ImageDataID}.nii.gz" (Subject before the underscore is the
  pairing key across SC and V04 folders).
- BraTS volumes are (240, 240, 155) and get padded to (240, 240, 160); ours
  are (91, 109, 91) and must reach (96, 128, 96) for the 4-level UNet to be
  safe after 3D Haar DWT (see _pad for why these numbers).
- We add degenerate-volume / shape-outlier filtering in __init__, because a
  handful of PPMI files are corrupted in ways that would produce NaNs during
  training (division by zero in normalize, or shape mismatch in DWT).
"""
import os
import os.path
import numpy as np
import nibabel
import torch
import torch.utils.data


# ---------------------------------------------------------------------------
# Intensity normalization (kept from bratsloader, with NaN-safety added)
# ---------------------------------------------------------------------------
def clip_and_normalize(img):
    """Clip to [0.1%, 99.9%] quantiles, then min-max scale to [0, 1].
    If the volume is degenerate (all-constant, so max==min), return zeros
    instead of dividing by zero."""
    q_lo = np.quantile(img, 0.001)
    q_hi = np.quantile(img, 0.999)
    img_clipped = np.clip(img, q_lo, q_hi)
    lo, hi = img_clipped.min(), img_clipped.max()
    denom = hi - lo
    if denom < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img_clipped - lo) / denom).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DaTSCANPairs(torch.utils.data.Dataset):
    """
    Expected directory layout:
        directory/
        |-- SC/      100001_I1452480.nii.gz,  100002_I1474774.nii.gz, ...
        `-- V04/     100001_I1530341.nii.gz,  100002_I1537899.nii.gz, ...

    Pairs are formed by the part of the filename before the first underscore
    (the PPMI Subject ID). Only subjects present in BOTH folders become pairs.
    """

    def __init__(self, directory, mode='train',
                 source_visit='SC', target_visit='V04',
                 expected_shape=(91, 109, 91),
                 target_shape=(96, 128, 96),
                 split_file=None, split_name='train'):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.source_visit = source_visit
        self.target_visit = target_visit
        self.expected_shape = expected_shape
        self.target_shape = target_shape

        # Index files by Subject ID (portion before first underscore)
        src_dir = os.path.join(self.directory, source_visit)
        tgt_dir = os.path.join(self.directory, target_visit)
        src_files = {f.split('_')[0]: os.path.join(src_dir, f)
                     for f in os.listdir(src_dir) if f.endswith('.nii.gz')}
        tgt_files = {f.split('_')[0]: os.path.join(tgt_dir, f)
                     for f in os.listdir(tgt_dir) if f.endswith('.nii.gz')}
        common_ids = sorted(set(src_files) & set(tgt_files))

        # ============================================================
        # [NEW] Split 필터링
        # ------------------------------------------------------------
        # split_file이 주어지면, JSON에서 split_name 키의 PID만 사용한다.
        # PID 단위 분할이므로 SC/V04 어느 쪽에 있든 무관 (둘 다 같은 PID).
        # split_file이 없으면 기존 동작(모든 paired PID 사용)을 유지한다.
        # ============================================================
        if split_file is not None:
            import json
            with open(split_file) as f:
                splits = json.load(f)
            if split_name not in splits:
                raise KeyError(f"'{split_name}' not in {split_file}. "
                               f"Available: {[k for k in splits if not k.startswith('_')]}")
            split_pids = set(str(p) for p in splits[split_name])
            before = len(common_ids)
            common_ids = [pid for pid in common_ids if pid in split_pids]
            print(f"[DaTSCANPairs] split='{split_name}': "
                  f"{len(common_ids)} of {before} paired PIDs kept")
            # split JSON에는 있지만 디스크에 없는 PID 경고 (전처리 누락 등 catch)
            missing = split_pids - set(common_ids)
            if missing:
                print(f"  [warning] {len(missing)} PIDs in split but not on disk")
        # ============================================================

        # Quality filter. Done once at __init__ so training never crashes mid-run.
        # get_fdata() reads the full volume (~4 MB each), ~6s total for 800 patients.
        self.database = []
        for pid in common_ids:
            try:
                src_img = nibabel.load(src_files[pid])
                tgt_img = nibabel.load(tgt_files[pid])
            except Exception as e:
                print(f'[skip] {pid}: cannot open ({e})')
                continue
            if src_img.shape != expected_shape or tgt_img.shape != expected_shape:
                print(f'[skip shape] {pid}: src={src_img.shape}, tgt={tgt_img.shape}')
                continue
            src_data = src_img.get_fdata()
            tgt_data = tgt_img.get_fdata()
            # "Degenerate" = volume is constant everywhere. Would cause div-by-zero
            # in clip_and_normalize and poison training with NaN.
            if (src_data.max() - src_data.min()) < 1e-8 or \
               (tgt_data.max() - tgt_data.min()) < 1e-8:
                print(f'[skip degenerate] {pid}')
                continue
            self.database.append({
                'patient_id': pid,
                'source': src_files[pid],
                'target': tgt_files[pid],
            })
        print(f'[DaTSCANPairs] paired samples: {len(self.database)} '
              f'({source_visit} -> {target_visit})')

    def _pad(self, vol_np):
        """Center-pad (91,109,91) -> (96,128,96) and add channel dim.

        Why these exact numbers:
          - We apply a 3D Haar DWT once -> spatial dims halved.
          - Then a 4-level UNet downsamples 3 more times (factor 8 more).
          - Total divisor: 2 * 8 = 16. So the input *must* be a multiple of 16
            along each axis, else decoder skip-connections have mismatched
            shapes and torch throws "Sizes of tensors must match" errors.
          - Smallest multiple of 16 that is >= original size:
                x: 91  -> 96   (+5)
                y: 109 -> 128  (+19)
                z: 91  -> 96   (+5)
          - 128 on y (instead of 112) is required because 112/16 = 7 which
            breaks the 4-level downsample (we hit this empirically).
        """
        out = torch.zeros(1, *self.target_shape)
        d0, d1, d2 = vol_np.shape
        t0, t1, t2 = self.target_shape
        # Center placement: evenly distribute the extra voxels on both sides
        # so the brain stays in the middle (convolutions rely on that spatial prior).
        s0 = (t0 - d0) // 2
        s1 = (t1 - d1) // 2
        s2 = (t2 - d2) // 2
        out[0, s0:s0 + d0, s1:s1 + d1, s2:s2 + d2] = torch.tensor(vol_np)
        return out

    def __getitem__(self, x):
        f = self.database[x]
        src_np = clip_and_normalize(nibabel.load(f['source']).get_fdata())
        tgt_np = clip_and_normalize(nibabel.load(f['target']).get_fdata())

        subj = f['source'] if self.mode in ('eval', 'auto') else 'dummy'

        return {
            'source':     self._pad(src_np).float(),   # SC, spatial condition
            'target':     self._pad(tgt_np).float(),   # V04, generation target
            'patient_id': f['patient_id'],
            'subj':       subj,
        }

    def __len__(self):
        return len(self.database)
