#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakengpu2
#SBATCH --job-name od1s
#SBATCH --time=10:00:00
source /scratch/algo/aitalla/stage-envPytorch/bin/activate
python /scratch/algo/aitalla/StageGitlab/src/training/train.py --expe_name Onlinedecaying1step --config /scratch/algo/aitalla/StageGitlab/configs/train/online/online1.yaml  $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out