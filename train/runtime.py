import random
import numpy as np
import torch


def initialize(device_name, seed, threads=4):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0' if device_name == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device_name == 'auto' else device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device
