#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<lg519> # required to send email notifcations - please replace <your_username> with your college login name or email address



export PATH=/vol/bitbucket/${USER}/HeartRateForecasting/HeartRateForecasting_env/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime


# use cuda 11.2
# if [ -f /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh ]
#     then
#         . /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
#     else
#         echo "CUDA 11.2.1 not found"
# fi

# run the python scripts
cd /vol/bitbucket/${USER}/HeartRateForecasting
python3 /vol/bitbucket/${USER}/HeartRateForecasting/heart_rate_dataset.py
python3 /vol/bitbucket/${USER}/HeartRateForecasting/LSTM_autoregressive_model.py