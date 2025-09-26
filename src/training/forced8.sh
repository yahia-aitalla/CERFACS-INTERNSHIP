#!/bin/bash
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakenepyc2
#SBATCH --job-name f8s
#SBATCH --time=16:00:00
source /scratch/algo/aitalla/stage-envPytorch/bin/activate
python /scratch/algo/aitalla/StageGitlab/src/training/train.py --expe_name 256forced8step --config /scratch/algo/aitalla/StageGitlab/configs/train/offline/forced8.yaml  $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out