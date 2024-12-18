'''
The following code is used to train the CORnet model with the self penalty term.
The self penalty term is added to the loss function to penalize the model for making a decision too late.
'''


import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from VGG_WW import NegativePearsonCorrelationLoss
from utils import set_seeds
from tqdm import tqdm
import os
import torch.optim as optim
import pandas as pd
import numpy as np
from dataset.data_loader import KarDataset
from torch.utils.data import DataLoader
from RNN_decision import DiffDecisionMultiClass, DiffDecision
import matplotlib.pyplot as plt
import scipy.stats as stats
from cornet import Flatten, Identity
from collections import OrderedDict


class cornet_wrapper(nn.Module):
    def __init__(self, time_steps=20, sigma=2.0):
        super(cornet_wrapper, self).__init__()
        self.time_steps = time_steps
        self.fc2 = nn.Linear(time_steps, 1, bias=False)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 10)),
            ('output', Identity())
        ]))
        self.w = nn.Parameter(torch.ones((1, time_steps)) * 0.1)
        self.b = nn.Parameter(torch.zeros((1, time_steps)))
        self.linear = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear1', nn.Linear(512, 64)),
            ('ReLU', nn.ReLU()),
            ('linear2', nn.Linear(64, 64)),
            ('ReLU', nn.ReLU()),
            ('linear3', nn.Linear(64, 10)),
            ('output', Identity())
        ]))
        self.threshold = torch.nn.Parameter(torch.tensor(1.00)) # threshold for the decision
        self.sigma = sigma

    def forward(self, hidden_state):

        s_traj = (self.linear(hidden_state.view(-1, 512, 7, 7)).view(-1, self.time_steps, 10).permute(0, 2, 1) * self.w + self.b).permute(0, 2, 1)
        s_accumulated = torch.cumsum(s_traj, dim=1)

        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)

        decision_time = DiffDecisionMultiClass.apply(s_accumulated - self.threshold, dsdt_trajectory)
        decision_time_min = decision_time.min(dim=1).values
        soft_index = torch.sigmoid((decision_time_min.unsqueeze(1) - torch.arange(self.time_steps, device=decision_time_min.device) + 0.5) * self.sigma)
        logit_trajectory = self.decoder(hidden_state.view(-1, 512, 7, 7)).view(-1, self.time_steps, 10).permute(0, 2, 1) * self.fc2.weight

        decision_logits = (logit_trajectory * soft_index.unsqueeze(1)).sum(dim=2)

        return decision_logits, decision_time_min


def train_model(model, optimizer, train_dataloader, test_dataloader, config, lr_scheduler=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_acc_list, epoch_loss_list, epoch_decision_time_list = [], [], []

    for epoch in tqdm(range(config['num_epochs'])):
        model.train()
        total_loss, total_acc, total_decision_time = 0.0, 0.0, 0.0

        for label, rt, rnn_preds, hidden_state in train_dataloader:

            optimizer.zero_grad()

            outputs, decision_time = model(hidden_state.to(device))
            loss = nn.CrossEntropyLoss()(outputs, label.to(device)) + (outputs[torch.arange(outputs.size(0)), label] * decision_time).mean() * config['lambda']
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            correct = outputs.argmax(dim=1).eq(label.to(device))
            running_acc = torch.sum(correct.float())
            running_loss = loss.item()

            total_loss += running_loss
            total_acc += running_acc.item()
            total_decision_time += decision_time.sum().item()

        epoch_acc_list.append(total_acc / len(train_dataloader.dataset))
        epoch_loss_list.append(total_loss)
        epoch_decision_time_list.append(total_decision_time / len(train_dataloader.dataset))

        # print the loss and acc
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Loss: {total_loss}")
            print(f"Epoch {epoch} Accuracy: {total_acc / len(train_dataloader.dataset)}")
            print(f"Epoch {epoch} Decision Time: {total_decision_time / len(train_dataloader.dataset)}")

            with torch.no_grad():
                evaluate_model(model, test_dataloader, config)
            model.train()

    print("Training complete!")

    # Test the model
    with torch.no_grad():
        fitted_model_rt, human_rt = evaluate_model(model, test_dataloader, config)

    os.makedirs(f"ckpt/{config['model_name']}", exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_acc_list': epoch_acc_list,
        'epoch_loss_list': epoch_loss_list,
        'epoch_decision_time_list': epoch_decision_time_list,
        'fitted_model_rt': fitted_model_rt,
        'human_rt': human_rt
    }
    torch.save(state, f"ckpt/{config['model_name']}/epoch_{epoch}.pth")


def evaluate_model(model, test_dataloader, config):
    model.eval()
    total_loss, total_acc, total_decision_time = 0.0, 0.0, 0.0
    human_rt, fitted_model_rt = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for label, rt, rnn_preds, hidden_state in test_dataloader:
        outputs, decision_time = model(hidden_state.to(device))
        human_rt.append(rt / 1000.0)
        fitted_model_rt.append(decision_time)
        loss = nn.CrossEntropyLoss()(outputs, label.to(device)) + (outputs[torch.arange(outputs.size(0)), label] * decision_time).mean() * config['lambda']
        correct = outputs.argmax(dim=1).eq(label.to(device))
        running_acc = torch.sum(correct.float())
        running_loss = loss.item()

        total_loss += running_loss
        total_acc += running_acc.item()
        total_decision_time += decision_time.sum().item()

    print(f"Test Loss: {total_loss}")
    print(f"Test Accuracy: {total_acc / len(test_dataloader.dataset)}")
    print(f"Test Decision Time: {total_decision_time / len(test_dataloader.dataset)}")

    # print the correlation between the model rt
    fitted_model_rt = torch.hstack(fitted_model_rt).detach().cpu().numpy()
    human_rt = torch.hstack(human_rt).detach().cpu().numpy()
    pearson_r, pearson_p = stats.pearsonr(fitted_model_rt, human_rt)
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.2f}")

    return fitted_model_rt, human_rt


def main():
    set_seeds()

    model = cornet_wrapper()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load('Kar_dataset/cornet.pth', map_location=device)
    for k in list(ckpt.keys()):
        if k.startswith('module.'):
            ckpt[k[7:]] = ckpt[k]
            del ckpt[k]

    model.decoder.linear.weight.data = ckpt['decoder.linear.weight']
    model.decoder.linear.bias.data = ckpt['decoder.linear.bias']
    model.fc2.weight.data = ckpt['time_weights.weight']
    for param in model.fc2.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    rt_df = pd.read_csv('Kar_dataset/human_rt_data.csv')
    hidden_states = np.load('Kar_dataset/hidden_states.npy', allow_pickle=True).item()
    train_hidden, test_hidden = hidden_states['train'], hidden_states['val']

    train_dict = {}
    test_dict = {}
    for key in train_hidden[0].keys():
        train_dict[key] = [d[key] for d in train_hidden]
        test_dict[key] = [d[key] for d in test_hidden]

    train_dataset = KarDataset(train_dict, rt_df, 'cornet')
    valid_dataset = KarDataset(test_dict, rt_df, 'cornet')

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)
    train_model(model, optimizer, train_loader, valid_loader,
                lr_scheduler=lr_scheduler, config={
            'num_epochs': 10000,
            'lambda': 0.0005,
            'model_name': f'cornet/self_penalty'})


# Train the model
if __name__ == '__main__':
    main()