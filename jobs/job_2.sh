#!/bin/bash

#SBATCH --job-name=haddock3
#SBATCH --array=0-239
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

TRAINS=(5 10 15 20)
TESTS=(1 2 3)
CS=(0.1 0.5 0.05 1.0)
MODELS=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D" "esm2_t33_650M_UR50D" "esm2_t36_3B_UR50D")

i=$SLURM_ARRAY_TASK_ID

train_idx=$(( i / (3 * 4 * 5) ))
test_idx=$(( (i / (4 * 5)) % 3 ))
c_idx=$(( (i / 5) % 4 ))
model_idx=$(( i % 5 ))

TRAIN=${TRAINS[$train_idx]}
TEST=${TESTS[$test_idx]}
C=${CS[$c_idx]}
MODEL=${MODELS[$model_idx]}

echo "Running job $SLURM_ARRAY_TASK_ID"
echo "TRAIN=$TRAIN TEST=$TEST C=$C MODEL=$MODEL"

python3 /d/hpc/home/mm5129/final_pipeline_cpu_only_embeddings_combine_n_with_c.py \
    --train "$TRAIN" \
    --test "$TEST" \
    --model "$MODEL" \
    --c "$C"
