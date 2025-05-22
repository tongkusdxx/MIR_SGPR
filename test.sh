#!/bin/bash
#SBATCH --job-name=test
#SBATCH -p gpu-rtx3080ti
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4

python test.py --st_root="../MIR_SGPR/image_for_test/st/" --de_root="../MIR_SGPR/image_for_test/gt/" --mask_root="../MIR_SGPR/image_for_test/mask/"
