"""
Sanity check before training. Run once and confirm:
  - The number of pairs matches expectations (~746 for current PPMI data).
  - A sample tensor has shape (1, 96, 128, 96) and range [0, 1].
  - patient_id parses correctly from filenames.

Usage:
    python scripts/prepare_data.py /workspace/DaTSCAN-nii
"""
import sys
sys.path.append('.')

from guided_diffusion.datscanloader import DaTSCANPairs


def main(data_dir):
    ds = DaTSCANPairs(data_dir, mode='train')
    print(f'\nTotal pairs: {len(ds)}')
    s = ds[0]
    print(f"source: shape={tuple(s['source'].shape)}, "
          f"range=[{s['source'].min():.3f}, {s['source'].max():.3f}]")
    print(f"target: shape={tuple(s['target'].shape)}, "
          f"range=[{s['target'].min():.3f}, {s['target'].max():.3f}]")
    print(f"patient_id: {s['patient_id']}")


if __name__ == '__main__':
    dd = sys.argv[1] if len(sys.argv) > 1 else '/workspace/DaTSCAN-nii'
    main(dd)
