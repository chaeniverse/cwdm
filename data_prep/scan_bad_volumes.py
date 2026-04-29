import numpy as np
import nibabel
import os

root = '/workspace/DaTSCAN-nii'
bad = []

for visit in ['SC', 'V04']:
    d = os.path.join(root, visit)
    for f in sorted(os.listdir(d)):
        if not f.endswith('.nii.gz'):
            continue
        path = os.path.join(d, f)
        try:
            img = nibabel.load(path).get_fdata()
        except Exception as e:
            bad.append((path, f'load_error: {e}'))
            continue

        q_lo = np.quantile(img, 0.001)
        q_hi = np.quantile(img, 0.999)
        clipped = np.clip(img, q_lo, q_hi)
        denom = clipped.max() - clipped.min()

        if denom < 1e-8:
            bad.append((path, f'degenerate: max-min={denom:.2e}, mean={img.mean():.3f}, nonzero={(img!=0).sum()}'))
        elif np.isnan(img).any():
            bad.append((path, 'contains NaN'))

print(f'\n=== {len(bad)} bad volumes found ===')
for p, reason in bad:
    print(f'  {os.path.basename(p):40s}  {reason}')
