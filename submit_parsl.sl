#!/bin/sh
#SBATCH --nodes=4
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH -p cpu_small


module load anaconda3/2020.11
module load gcc/8.3
module load cmake/3.23.2
conda activate /scratch/eubd/app/miniconda/envs/wazp
source /scratch/eubd/app/modules/elfutils-0.173-gcc8.sh
source /scratch/eubd/app/modules/libunwind-1.3.2-gcc8.sh

#acessa o diretório onde o script está localizado 
cd /scratch/eubd/carlos.cardoso2/PARSL/example4
python -m "pip install parsl==1.2.0"
python 1.py
