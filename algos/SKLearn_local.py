import numpy as np
import torch
import random
from utils.metrics import MSE, accuracy, F1
from data.synthetic import generate_data

# params
seeds = 