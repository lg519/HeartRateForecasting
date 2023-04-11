#!/bin/bash

# source .env file
source $(dirname $0)/.env

ssh -J $USERNAME@shell5.doc.ic.ac.uk $USERNAME@gpucluster2.doc.ic.ac.uk