# This script contains the implementation of the network and trains on random dot motion task
# Note that a similar implementation can be found in AlexNet_BN_LSTM_backbone_2.py
# The only difference is this version learned a temporal kernel that explicitly aggregates the outputs over time
# For replicating the results in main text, use this implementation
# But for replicating the results in SI, use AlexNet_BN_LSTM_backbone_2.py

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from utils import set_seeds
from train_backbone import train_model


class AlexNet_BN_LSTM(nn.Module):
    def __init__(self, time_steps=150, hidden_size=4096):
        super(AlexNet_BN_LSTM, self).__init__()
        self.time_steps = time_steps
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=256 * 1 * 1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(time_steps, 1)
        self.relu = nn.ReLU(inplace=True)
        self.output = False
        self.__init_weight()

    def forward(self, x):
        for t in range(self.time_steps):
            features = self.features(x[:, t, :, :, :]).squeeze(3).permute(0, 2, 1)
            if t == 0:
                features_all = features
            else:
                features_all = torch.cat((features_all, features), dim=1)
        lstm_out, _ = self.lstm(features_all)
        if self.output:
            return lstm_out
        else:
            out = self.fc1(lstm_out).squeeze()
            out = self.fc2(out)
            return out

    def __init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# Train the model
if __name__ == '__main__':
    set_seeds()
    
    model = AlexNet_BN_LSTM(time_steps=150)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 10 else 0.1)

    train_model(model, criterion, optimizer, num_epochs=200, model_name='AlexNet_BN_LSTM_norm', lr_scheduler=lr_scheduler, norm=True, curiculum=True)