"""
A script for sampling from a diffusion model for paired image-to-image translation.

=========================================================
PATCHED from original cwdm/scripts/sample.py for DaTScan SC -> V04.

Summary of changes (marked inline with "# [PATCHED]"):
  (1) Import DaTSCANPairs instead of BRATSVolumes.
  (2) Replace the 4-modality BraTS branching with a single source/target pair.
  (3) Condition is just SC (1 modality -> 8 wavelet channels), not 3 modalities (24 ch).
  (4) Noise tensor uses our wavelet-domain shape (48, 64, 48), not BraTS (112, 112, 80).
  (5) patient_id used as subject identifier (no 'validation/' path parsing).
  (6) Post-processing: center-crop back from (96,128,96) padded shape to original
      PPMI DaTScan shape (91,109,91); removed BraTS-specific brain-mask zero-out
      and the z<=155 crop.
=========================================================
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import (dist_util, logger)
from guided_diffusion.datscanloader import DaTSCANPairs      # [PATCHED] was bratsloader/BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    if args.dataset == 'brats':
        split_file = args.split_file if args.split_file else None
        ds = DaTSCANPairs(
            args.data_dir,
            mode='eval',                    # [FIX] sample이므로 eval
            split_file=split_file,
            split_name=args.split_name,     # [FIX] CLI 인자로 받은 split (val 또는 test)
        )


    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        # [PATCHED] Only two volumes in our batch dict: source (SC) and target (V04).
        batch['source'] = batch['source'].to(dist_util.dev())
        batch['target'] = batch['target'].to(dist_util.dev())

        # [PATCHED] patient_id comes directly from loader; no path slicing.
        subj = batch['patient_id'][0]
        print(subj)

        # [PATCHED] Single condition (SC), single target (V04). No 4-way contr branching.
        target = batch['target']     # V04 ground truth (for side-by-side saving)
        cond_1 = batch['source']     # SC baseline (the actual conditioning input)

        # [PATCHED] Conditioning vector: DWT one modality -> 8 subband channels.
        # (Original concatenated 3 modalities' DWTs for 24 channels.)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        # [PATCHED] Wavelet-domain noise shape for our input size:
        #   Original volume (padded): (96, 128, 96)
        #   After one DWT (halved):    (48, 64, 48)
        #   8 channels (L/H subbands).
        noise = th.randn(args.batch_size, 8, 48, 64, 48).to(dist_util.dev())

        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        # Inverse wavelet transform: 8 subband channels -> 1 image volume at (96,128,96).
        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        # Clamp to [0, 1] (our intensity normalization range).
        sample[sample <= 0] = 0
        sample[sample >= 1] = 1
        # [PATCHED] Original did: sample[cond_1 == 0] = 0  (brain mask zero-out).
        # BraTS is skull-stripped so 0 = background; DaTScan is NOT, so this
        # heuristic would wipe valid low-signal regions. Removed.

        # Drop channel dim: (B, 1, 96, 128, 96) -> (B, 96, 128, 96)
        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)
        if len(target.shape) == 5:
            target = target.squeeze(dim=1)

        # [PATCHED] Center-crop from padded (96, 128, 96) back to original
        # PPMI DaTScan shape (91, 109, 91). This mirrors datscanloader._pad:
        #   x: pad starts at 2 -> crop [2 : 2+91  = 93]
        #   y: pad starts at 9 -> crop [9 : 9+109 = 118]
        #   z: pad starts at 2 -> crop [2 : 2+91  = 93]
        # (Original BraTS code did a z[:155] crop; that shape doesn't apply here.)
        sample = sample[:, 2:93, 9:118, 2:93]
        target = target[:, 2:93, 9:118, 2:93]

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.output_dir, subj)).mkdir(parents=True, exist_ok=True)

        for i in range(sample.shape[0]):
            output_name = os.path.join(args.output_dir, subj, 'sample.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, subj, 'target.nii.gz')
            img = nib.Nifti1Image(target.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        split_file="",  # [NEW] split JSON 파일 경로
        split_name="test",  # [NEW] 사용할 split 이름 (val 또는 test)
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,
        contr="",
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
