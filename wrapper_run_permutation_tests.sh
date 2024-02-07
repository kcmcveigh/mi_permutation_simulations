#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=22:30:00
#SBATCH --job-name=permutation
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --partition=short
#SBATCH --array=50
#SBATCH --output=job_output/run_permutation_%j.out

module load anaconda3/2022.05
source activate KieranBase
cd /work/abslab/Kieran/CardioRespiratory/Code/mi_permutation_analyses/

srun python run_permutation_tests_parallel-retest.py $SLURM_ARRAY_TASK_ID
