#!/bin/bash

#SBATCH --job-name=haddock3_model1
#SBATCH --array=0-120          # 10 trains * 10 tests = 100 jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

# Use exact 1..50 by 5
TRAINS=(1 5 10 15 20 25 30 35 40 45 50)
TESTS=(1 5 10 15 20 25 30 35 40 45 50)
MODEL="esm2_t36_3B_UR50D"

num_trains=${#TRAINS[@]}
num_tests=${#TESTS[@]}

i=$SLURM_ARRAY_TASK_ID

train_idx=$(( i / num_tests ))  # changes every num_tests
test_idx=$(( i % num_tests ))   # cycles every 1

TRAIN=${TRAINS[$train_idx]}
TEST=${TESTS[$test_idx]}
# TRAIN=5
# TEST=2
echo "TRAIN=$TRAIN TEST=$TEST MODEL=$MODEL"
echo "Running job $SLURM_ARRAY_TASK_ID with train=$TRAIN, test=$TEST, model=$MODEL"

python3 /d/hpc/home/mm5129/final_pipeline_cpu_pca.py \
    --train "$TRAIN" \
    --test "$TEST" \
    #--model "$MODEL"
