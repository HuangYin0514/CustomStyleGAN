#!/bin/bash

rm -rf ./logs*

mv /mnt/c/Users/10713/Downloads/logs.tar ./logs.tar

tar zxf logs.tar

tensorboard --logdir=./
