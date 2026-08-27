#!/bin/bash

#SBATCH --job-name=haddock3      # job name
#SBATCH --array=0-11          # 6 jobs (index 0–5)
#SBATCH --nodes=1                # use 1 node
#SBATCH --ntasks=1               # 1 task (your python script)
#SBATCH --cpus-per-task=10

#SBATCH --time 06:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate

TRAINS=(5 10 15 20)
#TRAINS=(25 30 35 40)

TESTS=(1 2 3)
CS=(0.1)

i=$SLURM_ARRAY_TASK_ID

train_idx=$(( i / (3*1) ))   # 0..3 (changes every 3)
test_idx=$(( (i / 1) % 3 ))  # 0..2 (cycles every 1)
c_idx=$(( i % 1 ))           # always 0

TRAIN=${TRAINS[$train_idx]}
TEST=${TESTS[$test_idx]}
C=${CS[$c_idx]}

echo "TRAIN=$TRAIN TEST=$TEST C=$C"
echo "Running job $SLURM_ARRAY_TASK_ID with train=$TRAIN, test=$TEST, c=$C"

python3 /d/hpc/home/mm5129/final_pipeline_cpu.py \
    --train "$TRAIN" \
    --test "$TEST" \
    --model "prot_t5"
    #--c "$C" \
    