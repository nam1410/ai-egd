#!/bin/bash
#SBATCH --job-name=dinov2_kfold
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-4   # 5 folds (adjust range if folds differ)

module --force purge
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/12.6 opencv/4.12

source ~/venvs/ai_egd/bin/activate

# FOLD_INDEX=${SLURM_ARRAY_TASK_ID}

python -m src.train_dinov2_single_fold \
  --data_root /lustre06/project/6103394/ofarooq/AIEGD_datasets/ \
  --patient_config_json /lustre06/project/6103394/ofarooq/ai-egd/src/datasets/patient_config_ann_missing.json \
  --fold_index 4 \
  --folds 5 \
  --epochs 100 \
  --lr 0.00001 \
  --save_dir ./test_checkpoints_dinov2_kfold \
  --splits_file ./test_checkpoints_dinov2_kfold/kfold_splits.npz

# After array completes
python -m src.eval_kfold_results \
    --data_root /lustre06/project/6103394/ofarooq/AIEGD_datasets/ \
    --patient_config_json /lustre06/project/6103394/ofarooq/ai-egd/src/datasets/patient_config_ann_missing.json \
    --folds 5 \
    --splits_file ./test_checkpoints_dinov2_kfold/kfold_splits.npz \
    --save_dir ./test_checkpoints_dinov2_kfold