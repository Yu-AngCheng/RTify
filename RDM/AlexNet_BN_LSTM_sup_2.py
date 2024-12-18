"""
The following script trains the RTify module with human RT data on the model.
For self penalty, use the snippet from RDM/AlexNet_BN_LSTM_self_sup_2.py
The following script replicates the results in the SI.
For replicating the results in the main text, use the snippet from RDM/AlexNet_BN_LSTM_sup.py
"""

import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seeds
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import SoftHistogram
from utils import load_human_rt_dataset
import torch.nn.functional as F
from RNN_decision import DiffDecision


class AlexNet_BN_LSTM_wrapper(nn.Module):
    def __init__(self, time_steps=150, dt=1/75.0, sigma=2.0):
        super(AlexNet_BN_LSTM_wrapper, self).__init__()
        self.time_steps = time_steps
        self.fc1 = nn.Linear(4096, 1)
        self.linear = nn.Linear(4096, 1)
        self.w = nn.Parameter(torch.ones((1, 1)))
        self.b = nn.Parameter(torch.zeros((1, 1)))
        self.threshold = torch.nn.Parameter(torch.tensor(1.0)) # threshold for the decision
        self.dt = dt
        self.sigma = sigma

    def forward(self, lstm_out):
        s_traj = self.linear(lstm_out).squeeze() * self.w + self.b
        s_accumulated = torch.cumsum(s_traj, dim=1)
        s_traj_abs = torch.abs(s_accumulated)

        dsdt_trajectory = torch.diff(s_traj_abs, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)

        decision_time = DiffDecision.apply(s_traj_abs - self.threshold, dsdt_trajectory)
        soft_index = torch.exp(-0.5 * (decision_time.unsqueeze(1) - torch.arange(self.time_steps,  device=decision_time.device)) ** 2 / self.sigma ** 2)
        soft_index = soft_index / soft_index.sum(dim=1).unsqueeze(1)
        logit_trajectory = self.fc1(lstm_out).squeeze()
        decision_logits = (logit_trajectory * soft_index).sum(dim=1, keepdim=True)

        return decision_logits, decision_time


def train_model(model, optimizer, histogram, human_rt_dist_train, human_rt_dist_test, config, coherence_list=None, lr_scheduler=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_outputs_list, direction_list = torch.load(config['outputs_path'])
    backbone_outputs_list = torch.stack(backbone_outputs_list).to(device)

    if coherence_list is None:
        coherence_list = [51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]

    signed_RT_coh_list = [[] for _ in range(len(coherence_list))]
    epoch_acc_list = []
    epoch_loss_list = []
    epoch_decision_time_list = []

    for epoch in tqdm(range(config['num_epochs'])):
        total_loss = 0.0
        total_acc = 0.0
        total_decision_time = 0.0

        for i_coh, coh in enumerate(coherence_list):

            indices = torch.randperm(config['dataset_size'], device='cpu')[:config['BATCH_SIZE']]

            optimizer.zero_grad()
            outputs, decision_time = model(backbone_outputs_list[i_coh][indices].to(device))
            correct = ((outputs > 0) == (direction_list[i_coh][indices].to(device) > 0)).squeeze()
            loss = F.mse_loss(histogram(decision_time * model.dt * (2*correct.float()-1)), human_rt_dist_train[i_coh].to(device))

            loss.backward()
            optimizer.step()

            correct = ((outputs > 0) == (direction_list[i_coh][indices].to(device) > 0)).squeeze()
            coherence_acc = torch.sum(correct.float()) / config['BATCH_SIZE']
            coherence_loss = loss
            signed_RT_coh_list[i_coh].append((decision_time * (2*correct.float() - 1)).detach().cpu().numpy())

            total_loss += coherence_loss.item()
            total_acc += coherence_acc.item()
            total_decision_time += decision_time.mean().item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_loss_list.append(total_loss/len(coherence_list))
        epoch_acc_list.append(total_acc/len(coherence_list))
        epoch_decision_time_list.append(total_decision_time/len(coherence_list))

    os.makedirs(f"ckpt/{config['model_name']}", exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_acc_list': epoch_acc_list,
        'epoch_loss_list': epoch_loss_list,
        'epoch_decision_time_list': epoch_decision_time_list,
        'signed_RT_coh_list': signed_RT_coh_list,
    }
    torch.save(state, f"ckpt/{config['model_name']}/epoch_{epoch}.pth")

    print("Training complete!")

    # plotting usage, uncomment to plot the results

    # histogram.to(torch.device("cpu"))
    # # plot the histogram of the signed RT
    # fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    # ax = axs.flatten()
    # for i in range(len(coherence_list)):
    #     model_rt = torch.tensor(np.array(signed_RT_coh_list[i][-10:])).flatten()
    #     ax[i].plot(histogram.centers, histogram(model_rt * model.dt).detach().cpu().numpy())
    #     ax[i].plot(histogram.centers, human_rt_dist_test[i].cpu().numpy())
    #     ax[i].set_title(f'Coherence {coherence_list[i]} Signed RT histogram')
    #     ax[i].set_xlabel('Signed RT')
    #     ax[i].set_ylabel('Frequency')
    #     ax[i].set_xlim(-2.5, 2.5)
    # plt.tight_layout()
    # plt.savefig(f"ckpt/{config['model_name']}/signed_RT_histogram.png")


def main():
    set_seeds()

    model = AlexNet_BN_LSTM_wrapper()
    optimizer = optim.Adam(model.linear.parameters(), lr=1e-6)
    optimizer.add_param_group({'params': model.w, 'lr': 1e-6})
    optimizer.add_param_group({'params': model.b, 'lr': 1e-6})
    optimizer.add_param_group({'params': model.threshold, 'lr': 2*1e-4})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('ckpt/AlexNet_BN_LSTM_norm_2/epoch_100.pth', map_location=device)['state_dict'], strict=False)

    for param in model.fc1.parameters():
        param.requires_grad = False

    histogram = SoftHistogram(500, -2.5, 2.5, 40.0).to(device)
    human_rt_dist_train, human_rt_dist_test = load_human_rt_dataset('../dataset/human_rt_dataset.pkl', [0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2], histogram)
    # flip the first dimension of the human_rt_dist_train and human_rt_dist_test，since the highest coherence is the first row
    human_rt_dist_train, human_rt_dist_test  = human_rt_dist_train.flip(0), human_rt_dist_test.flip(0)

    train_model(model, optimizer, histogram, human_rt_dist_train, human_rt_dist_test,
                lr_scheduler=None, config={
                'num_epochs': 10000,
                'BATCH_SIZE': 1024,
                'dataset_size': 2048,
                'model_name': f'AlexNet_BN_LSTM_BP_2/human_fit',
                'outputs_path': 'ckpt/AlexNet_BN_LSTM_norm_2/backbone_outputs_epoch_100_dynamic.pth'})


# Train the model
if __name__ == '__main__':
    main()



