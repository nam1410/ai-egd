#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out   # optional: capture output

# Load any necessary modules (if required by your cluster)
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/12.6 opencv/4.12

# Activate virtual environment
source ~/venvs/ai_egd/bin/activate


# Check GPU visibility (for debugging)
nvidia-smi

# to check if there are any errors in the script
python src/datasets/endo_xml_classification.py
