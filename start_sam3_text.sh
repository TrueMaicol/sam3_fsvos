#!/bin/bash

# # $1 = exp name
# # $2 = fold
# # $3 = seed
# # $4 = run number

#SBATCH --time=10:00:00                         # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                               # Number of nodes to use
#SBATCH --ntasks-per-node=1                     # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                            # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=iscrc_marsv2                  # Project account number

# Load necessary modules (adjust to your environment)
# module load cuda/12.6

# source /leonardo/home/userexternal/mcavicch/miniconda3/etc/profile.d/conda.sh
# conda activate ./sam3_venv

module load cuda/12.6

source /leonardo/home/userexternal/mcavicch/miniconda3/etc/profile.d/conda.sh
conda activate ./sam3_venv


echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

cd src

export HF_HOME="/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/models"
export TRANSFORMERS_CACHE="/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/models"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "$4" ]; then
    run_dir="FULL_TEST_$4"
else
    run_dir="FULL_TEST_1"
fi

if [ -n "$5" ]; then
    N_SHOT=$5
else
    N_SHOT=5
fi

srun python test_SAM3_FSVOS.py \
    --checkpoint /leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/checkpoints/sam3.pt \
    --vlm_model_path /leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/models \
    --group ${2} \
    --dataset_path /leonardo_work/IscrC_MARSv2/datasets/Youtube-FSVOS/train \
    --output_dir /leonardo_scratch/large/userexternal/mcavicch/SAM3_OUTPUT_DATA/${1}/${run_dir} \
    --session_name fold_${2} \
    --seed ${3} \
    --nshot ${N_SHOT} \
    --text_prompt \
    --use_support_visuals 