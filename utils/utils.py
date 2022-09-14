import pickle
import random

import numpy as np
import torch


def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

def would_user_further_response():
    return False

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
