#!/bin/bash

python main.py \
  --model linreg \
  --dataset synthetic \
  --n_clients 100\
  --algo persfl \
  --R 1500 \
  --lrate=0.03\
  --reps=10\
  --seed 0
