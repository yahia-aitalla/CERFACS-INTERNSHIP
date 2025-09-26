#!/bin/bash
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakenepyc2
#SBATCH --job-name of8s
#SBATCH --time=10:00:00
source /scratch/algo/aitalla/stage-envPytorch/bin/activate
cd /scratch/algo/aitalla/StageGitlab/src
python -m training.train --expe_name test_online_8forced --config /scratch/algo/aitalla/StageGitlab/configs/train/online/online.yaml  $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out