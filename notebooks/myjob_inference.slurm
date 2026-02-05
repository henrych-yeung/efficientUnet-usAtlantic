#!/bin/bash
 
#Submit this script with: sbatch myjob.slurm
 
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --constraint="a100|v100"
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=apm5rt@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=sds_capstone_atashman
#SBATCH --output=logs/%x_%A.out
 
#### Run a single script ####
# The script randomly shuffles the input directory to reduce overlap. Simply run multiple slurm jobs with this script

cd ~/efficientUnet-usAtlantic/notebooks
module load apptainer
apptainer exec --nv ../keras-rasterio_latest.sif \
  python 3-RasterAnalysis.py \
  --input "/scratch/apm5rt/NAIP_1m" \
  --output "/scratch/apm5rt/segmentation_1m" \
  --stable_min 3