#!/bin/bash
#SBATCH -o /home/knuth/thesis/Implementation/FederatedLearning/logs.%j.%N.log
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node 4
#SBATCH --mem=8000
#SBATCH --array 1-29


python3 Local_Trainers.py $SLURM_JOB_ID
