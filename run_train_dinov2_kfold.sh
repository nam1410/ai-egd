#!/bin/bash
#SBATCH --job-name=dinov2_endo_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00              # Adjust based on your needs
#SBATCH --mem=16G                    # Memory per node
#SBATCH --cpus-per-task=4            # CPU cores for data loading
#SBATCH --gres=gpu:1                 # Request 1 GPU


module --force purge
# Load any necessary modules (if required by your cluster)
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/12.6 opencv/4.12


# Activate virtual environment
source ~/venvs/ai_egd/bin/activate


# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
echo "GPU info:"
nvidia-smi

# Run training

python -m src.train_dinov2_kfold \
    --data_root /lustre06/project/6103394/ofarooq/AIEGD_datasets/ \
    --patient_config_json /lustre06/project/6103394/ofarooq/ai-egd/src/datasets/patient_config_ann_missing.json \
    --epochs 100 \
    --lr 0.00001 \
    --save_dir ./test_checkpoints_dinov2_kfold

echo "End time: $(date)"
echo "Job completed with exit code: $?"
