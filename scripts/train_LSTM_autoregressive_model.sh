#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<lg519> # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/HeartRateForecasting/HeartRateForecasting_env/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
export PATH=/vol/bitbucket/${USER}/HeartRateForecasting/:$PATH
python3 /vol/bitbucket/${USER}/HeartRateForecasting/LSTM_autoregressive_model.py