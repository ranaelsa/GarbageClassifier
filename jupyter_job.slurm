#! /bin/bash
# ====================================
#SBATCH --job-name=test_predictions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=jupyter_predictions.out
# ====================================

source ~/software/init-conda
conda activate pytorch

module load cuda

jupyter notebook --no-browser --port=8888 --ip=0.0.0.0