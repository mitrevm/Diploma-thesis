#!/bin/bash

#SBATCH --job-name=haddock3      # job name
#SBATCH --array=0-27           # 6 jobs (index 0–5)
#SBATCH --nodes=1                # use 1 node
#SBATCH --ntasks=1               # 1 task (your python script)
#SBATCH --cpus-per-task=10

#SBATCH --time 06:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

TRAINS=(5 10 15 20 21 22 23 24 25 26 27 28 29 30)   # 14 values
TESTS=(4 5)                                         # 2 values
CS=(0.1)  

i=$SLURM_ARRAY_TASK_ID

train_idx=$(( i / (2*1) ))     # 0..13  (changes every 2)
test_idx=$(( (i / 1) % 2 ))    # 0..1   (cycles every 1)
c_idx=$(( i % 1 ))             # always 0

TRAIN=${TRAINS[$train_idx]}
TEST=${TESTS[$test_idx]}
C=${CS[$c_idx]}

echo "Running job $SLURM_ARRAY_TASK_ID with train=$TRAIN, test=$TEST, c=$C"

python3 /d/hpc/home/mm5129/final_pipeline_cpu.py \
    --train "$TRAIN" \
    --test "$TEST" \
    --c "$C" \
    --model "esm2_t12_35M_UR50D"