import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from wong_wang import WongWangMultiClassDecision
import numpy as np
import os
import matplotlib
from utils import set_seeds
matplotlib.use('Agg')

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from dataset.data_loader import KarDataset_logits
from torch.utils.data import DataLoader
import scipy.stats as stats


class NegativePearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = -torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost


class WW_multiclass_wrapper(nn.Module):
    def __init__(self, n_classes=10, dt=10):
        super(WW_multiclass_wrapper, self).__init__()
        self.dt = dt
        self.linear = nn.Linear(n_classes, n_classes)
        self.WW = WongWangMultiClassDecision(n_classes=n_classes, dt=dt)
        self.linear.weight.data = torch.eye(n_classes) * 0.2
        self.linear.bias.data = torch.zeros(n_classes)

    def forward(self, model_name, logits, labels):
        x = torch.abs(self.linear(logits))
        decision_times = self.WW(x)
        return decision_times


def RT_fitting(model_name, rt_fitter, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(f'Kar_dataset/train_logits_{model_name}1.csv')
    val_df = pd.read_csv(f'Kar_dataset/test_logits_{model_name}1.csv')
    rt_df = pd.read_csv('Kar_dataset/human_rt_data.csv')

    train_dataset = KarDataset_logits(train_df, rt_df, model_name)
    valid_dataset = KarDataset_logits(val_df, rt_df, model_name)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    model = WW_multiclass_wrapper().to(device)

    model_parameters = {name: [] for name, p in model.named_parameters() if p.requires_grad}
    parameters_with_grad = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    loss_curve = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = NegativePearsonCorrelationLoss()

    for epoch in tqdm(range(config['num_epochs'])):

        total_loss = 0.0
        optimizer.zero_grad()
        for i, (labels, rt, logits) in enumerate(train_loader):
            labels, rt, logits = labels.to(device), rt.to(device), logits.to(device)
            if logits.dim() == 3:
                logits = logits.permute(0, 2, 1)

            decision_times_class = model(model_name, logits, labels)
            decisions_time = decision_times_class.min(dim=1).values

            loss = criterion(decisions_time, rt/1000.0)
            loss.backward()
            total_loss += loss.item() * len(labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        optimizer.step()
        loss_curve.append(total_loss / len(train_dataset))

        for name, p in parameters_with_grad:
            model_parameters[name].append(p.clone().detach().cpu())

    for name, p in parameters_with_grad:
        model_parameters[name] = torch.stack(model_parameters[name])

    model.eval()
    fitted_model_rt = []
    human_rt = []
    val_label = []
    for i, (labels, rt, logits) in enumerate(valid_loader):
        val_label.extend(labels.clone().detach().cpu())
        labels, rt, logits = labels.to(device), rt.to(device), logits.to(device)
        if logits.dim() == 3:
            logits = logits.permute(0, 2, 1)
        decision_time_sum = torch.zeros(rt.shape[0], device=device)
        for j in range(10):
            decision_times_class = model(model_name, logits, labels)
            decisions_time = decision_times_class.min(dim=1).values
            decision_time_sum += decisions_time
        fitted_model_rt.extend((decision_time_sum/10.0).squeeze().clone().detach().cpu())
        human_rt.extend(rt.clone().detach().cpu()/1000.0)
    fitted_model_rt = torch.hstack(fitted_model_rt).numpy()
    human_rt = torch.hstack(human_rt).numpy()
    val_label = torch.hstack(val_label).numpy()

    # Calculate and display the Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(fitted_model_rt, human_rt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(fitted_model_rt, human_rt)


    os.makedirs(f'ckpt/{model_name}/{rt_fitter}', exist_ok=True)
    # Plot the loss curve
    plt.figure()
    plt.plot(loss_curve, label=loss_curve[-1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_loss_curve.png')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(fitted_model_rt, human_rt, s=1)
    line = slope * np.array(fitted_model_rt) + intercept
    ax.plot(fitted_model_rt, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
    ax.set_xlabel('Model RT (a.u.)')
    ax.set_ylabel('Human RT (s)')
    ax.set_title(f'Pearson r={pearson_r:.2f}, p={pearson_p:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_fitted_RT.png')

    state = {
        'model': model_parameters,
        'loss_curve': loss_curve,
        'fitted_model_rt': fitted_model_rt,
        'human_rt': human_rt,
    }
    torch.save(state, f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_fit.pth')


if __name__ == "__main__":
    set_seeds(42)

    config = {
        'BATCH_SIZE': 1024,
        'num_epochs': 100000,
        'lr': 1e-4,
    }
    RT_fitting('vgg', 'WW_multiclass', config=config)

