#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<lg519> # required to send email notifcations - please replace <your_username> with your college login name or email address

set -e

export PATH=/vol/bitbucket/${USER}/HeartRateForecasting/HeartRateForecasting_env/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime


# use cuda 11.4
. /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

# run the python scripts
cd /vol/bitbucket/${USER}/HeartRateForecasting
python3 /vol/bitbucket/${USER}/HeartRateForecasting/heart_rate_dataset.py
python3 /vol/bitbucket/${USER}/HeartRateForecasting/LSTM_autoregressive_model.py