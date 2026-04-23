# =============================================================================
# cWDM run.sh  — PATCHED for DaTScan SC -> V04 longitudinal synthesis
# =============================================================================
# Differences from original BraTS run.sh:
#   - DATA_DIR points to our PPMI DaTScan pair folder
#   - IN_CHANNELS=16 (target 8 + single condition 8), was 32 (8 + 3*8)
#   - IMAGE_SIZE reflects padded volume size, not BraTS (224)
#   - CONTR is left in for arg compatibility but has no effect (no 4-way branching)
# =============================================================================

# ---- general settings -------------------------------------------------------
GPU=0
SEED=42
CHANNELS=64
MODE='sample'               # 'train' or 'sample'
DATASET='brats'            # kept as 'brats' so the dataset-name branches work unchanged
MODEL='unet'
CONTR='v04'                # [PATCHED] unused in our i2i pipeline; any string is fine

# ---- sampling settings (only used when MODE=sample) -------------------------
ITERATIONS=50              # which checkpoint to load (50k step = 50)
SAMPLING_STEPS=0           # 0 = full 1000-step reverse diffusion
RUN_DIR="runs/Apr19_12-39-37_559bd69da4ff"                 # set before sampling, e.g. runs/<timestamp>

# ---- model architecture (cWDM U-Net, 4-level downsample) --------------------
if [[ $MODEL == 'unet' ]]; then
  echo "MODEL: WDM (U-Net)"
  CHANNEL_MULT=1,2,2,4,4
  ADDITIVE_SKIP=False
  BATCH_SIZE=1
  IMAGE_SIZE=128           # [PATCHED] padded-volume reference size (mostly cosmetic)
  IN_CHANNELS=16           # [PATCHED] target(8) + condition(8) = 16. Was 32 for BraTS.
  NOISE_SCHED='linear'
else
  echo "MODEL TYPE NOT FOUND"
fi

# ---- data dir resolution ----------------------------------------------------
if [[ $MODE == 'train' ]]; then
  echo "MODE: training"
  DATA_DIR=/workspace/DaTSCAN-nii              # [PATCHED]
elif [[ $MODE == 'sample' ]]; then
  BATCH_SIZE=1
  echo "MODE: sampling (image-to-image translation)"
  DATA_DIR=/workspace/DaTSCAN-nii              # [PATCHED]
fi

# ---- shared flags -----------------------------------------------------------
COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=${NOISE_SCHED}
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=False
--predict_xstart=True
--contr=${CONTR}
"

# ---- train-only flags -------------------------------------------------------
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=5000
--lr_anneal_steps=50000
--num_workers=12
--devices=${GPU}
"

# ---- sample-only flags ------------------------------------------------------
SAMPLE="
--data_dir=${DATA_DIR}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--devices=${GPU}
--output_dir=./results/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"

# ---- dispatch ---------------------------------------------------------------
if   [[ $MODE == 'train'  ]]; then python scripts/train.py  $TRAIN  $COMMON
elif [[ $MODE == 'sample' ]]; then python scripts/sample.py $SAMPLE $COMMON
else echo "MODE NOT FOUND"
fi
