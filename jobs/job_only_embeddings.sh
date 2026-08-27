#!/bin/bash

#SBATCH --job-name=pca_sweep
#SBATCH --array=0-8               # We have 8 values in our list
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00           # PCA is fast, 2 hours is plenty
#SBATCH --output=pca_logs_%a.out

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

# Define the list of PCA components
PCA_VALUES=(2 3 5 8 10 12 15 20 25)

# Get the value for THIS specific array task
CURRENT_PCA=${PCA_VALUES[$SLURM_ARRAY_TASK_ID]}

MODEL="prot_albert"

echo "Running model $MODEL with PCA components: $CURRENT_PCA"

python3 /d/hpc/home/mm5129/final_pipeline_cpu_only_embeddings_pca.py \
    --train 1 \
    --test 1 \
    --model "$MODEL" \
    --pca "$CURRENT_PCA"