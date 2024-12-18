import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        delta = float(max - min) / float(bins)
        centers = float(min) + delta * (torch.arange(bins).float() + 0.5)
        self.register_buffer('centers', centers)
        self.register_buffer('delta', torch.tensor(delta))

    def forward(self, x):
        x = x.unsqueeze(0) - self.centers.unsqueeze(1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x / (x.sum() + 1e-6)


# The only difference between SoftHistogram_v2 and SoftHistogram is that SoftHistogram_v2 only consider x between min and max
# This helps to prevent the model learning from negative RTs, which implicitly leaks the accuracy information
class SoftHistogram_v2(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram_v2, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        delta = float(max - min) / float(bins)
        centers = float(min) + delta * (torch.arange(bins).float() + 0.5)
        self.register_buffer('centers', centers)
        self.register_buffer('delta', torch.tensor(delta))

    def forward(self, x):
        # only consider x between min and max
        x = x[(x >= self.min) & (x <= self.max)]
        x = x.unsqueeze(0) - self.centers.unsqueeze(1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x / (x.sum() + 1e-6)


def load_human_rt_dataset(file_path, coherence_list, histogram):
    with open(file_path, 'rb') as f:
        human_rt_train, human_rt_test = pickle.load(f)

    device = histogram.centers.device
    human_rt_dist_train = [[] for _ in range(len(coherence_list))]
    human_rt_dist_test = [[] for _ in range(len(coherence_list))]
    for i in range(len(coherence_list)):
        human_rt_dist_train[i] = histogram(torch.tensor(human_rt_train[i], dtype=torch.float32, device=device))
        human_rt_dist_test[i] = histogram(torch.tensor(human_rt_test[i], dtype=torch.float32, device=device))
    human_rt_dist_train = torch.stack(human_rt_dist_train)
    human_rt_dist_test = torch.stack(human_rt_dist_test)

    return human_rt_dist_train, human_rt_dist_test


