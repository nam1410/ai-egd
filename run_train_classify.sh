#!/bin/bash
#SBATCH --job-name=test_dinov2
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --time=00:30:00              # 30 minutes for testing
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module --force purge
# Load any necessary modules (if required by your cluster)
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/12.6 opencv/4.12

# Activate virtual environment
source ~/venvs/ai_egd/bin/activate

# Check GPU visibility (for debugging)
nvidia-smi


# Test with smaller epochs
python -m src.train_classify.py \
    --data_root /lustre06/project/6103394/ofarooq/AIEGD_datasets/ \
    --patient_config_json /lustre06/project/6103394/ofarooq/ai-egd/src/datasets/patient_config.json \
    --batch_size 8 \
    --epochs_stage1 1 \
    --epochs_stage2 1 \
    --num_workers 2 \
    --save_dir ./test_checkpoints