#!/bin/bash
#SBATCH --ntasks 16
#SBATCH --mem 200G
#SBATCH --time 4-0
#SBATCH --qos bham
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL

set -e

module purge
module load baskerville
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.1.1-GCC-10.2.0

export VENV_DIR="/bask/projects/c/changhj-train-dnn/virtual-environments"
export VENV_PROJECT="srf"
export VENV_PATH="${VENV_DIR}/${VENV_PROJECT}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    python -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# pip install package
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r pip_install.txt

# bash download_example_data.sh
python run_nerf.py --config configs/fern.txt    