#!/bin/bash
#SBATCH --job-name=transformer_training          # Job name
#SBATCH --output=output_%j.txt                   # Output file
#SBATCH --error=error_%j.txt                     # Error file
#SBATCH --time=48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                             # Number of GPUs per node
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=1                      # Number of tasks per node
#SBATCH --partition=gpu                          # Partition (queue) to submit to
#SBATCH --account=hpc_llm_mech           # Allocation name

# Load necessary modules
module load python/3.9.7-anaconda

source .llmEnv/bin/activate

export HF_HOME=/work/elo/huggingface_cache
export TRANSFORMERS_CACHE=/work/elo/huggingface_cache

# Run the Python script
python3 transformer_trainer.py --neptune_logger
