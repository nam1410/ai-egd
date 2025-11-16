#!/bin/bash
#SBATCH --output=slurm-%j.out   # optional: capture output

# Load any necessary modules (if required by your cluster)
# On login node (has internet)
module load StdEnv/2023 python/3.11.5
source ~/venvs/ai_egd/bin/activate
python download_dinov2.py