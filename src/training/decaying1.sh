#!/bin/bash
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakenepyc1
#SBATCH --job-name d1s
#SBATCH --time=16:00:00
source /scratch/algo/aitalla/stage-envPytorch/bin/activate
python /scratch/algo/aitalla/StageGitlab/src/training/train.py --expe_name 256decaying1step --config /scratch/algo/aitalla/StageGitlab/configs/train/offline/decaying1.yaml  $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out