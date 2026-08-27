#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name haddock3
#SBATCH --tasks-per-node=50
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time 04:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate
boltz predict /d/hpc/home/mm5129/C0JMJ0.fasta