#!/bin/bash
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=krakenepyc1
#SBATCH --job-name gendeca
#SBATCH --time=16:00:00
source /scratch/algo/aitalla/stage-env/bin/activate
python  /scratch/algo/aitalla/StageGitlab/src/datagen/generate.py --expe_name deca --config /scratch/algo/aitalla/StageGitlab/configs/data/decaying/decaying.yaml $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out