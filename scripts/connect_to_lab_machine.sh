#!/bin/bash

# source .env file
source $(dirname $0)/.env

# connect to lab machine
ssh $USERNAME@shell1.doc.ic.ac.uk

# connect to lab machine
/vol/linux/bin/sshtolab

# clone the repo
git clone https://github.com/lg519/HeartRateForecasting.git

# change directory
cd HeartRateForecasting

# run LSTM_autoregressive_model.py
python LSTM_autoregressive_model.py