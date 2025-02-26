#! /bin/bash
# ====================================
#SBATCH --job-name=garbage-classifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=output_log.out
# ====================================

source ~/software/init-conda
conda activate pytorch

python /home/rana.elsadig/enel645-a2/GarbageClassifier.py