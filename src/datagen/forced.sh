#!/bin/bash
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakenepyc1
#SBATCH --job-name g1
#SBATCH --time=16:00:00
source /scratch/algo/aitalla/stage-env/bin/activate
python  /scratch/algo/aitalla/StageGitlab/src/datagen/generate.py --config /scratch/algo/aitalla/StageGitlab/configs/data/forced/forced.yaml $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out