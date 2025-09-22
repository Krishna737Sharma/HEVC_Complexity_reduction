#!/bin/bash
#SBATCH --job-name=Vit-Model          # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=15gb                      # Job memory request
#SBATCH --cpus-per-task=20                # Number of CPU cores per task
#SBATCH --gpus-per-node=1               # Number of GPU
#SBATCH --partition=gpupart_p100              # Time limit hrs:min:sec
#SBATCH --time=1-23:50:00             # Time limit hrs:min:sec
#SBATCH --output=vit_training_op_26_march.log        # Standard output and error log

source /home/m1/23CS60R16/rishab_venv/bin/activate

python3 /home/m1/23CS60R16/Dec_6_Model_data/Hevc_models_script/ViT_2.3M/Vit_model.py