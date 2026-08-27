#!/bin/bash

#SBATCH --job-name=pca_sweep             # We have 8 values in our list
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00           # PCA is fast, 2 hours is plenty
#SBATCH --output=pca_logs_%a.out

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

# Define the list of PCA components

# Get the value for THIS specific array task

MODEL="prot_albert"


python3 /d/hpc/home/mm5129/final_pipeline_cpu_only_embeddings_pca.py \
    --train 1 \
    --test 1 \
    --model "$MODEL" \