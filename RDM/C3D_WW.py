import torch
import torch.nn as nn
import torch.nn.functional as F
from wong_wang import WongWangDecisionBP
from utils import SoftHistogram
from utils import load_human_rt_dataset
import numpy as np
import os
import matplotlib
from utils import set_seeds
matplotlib.use('Agg')
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


class WW_wrapper(nn.Module):
    def __init__(self, histogram, dt=1000.0/75.0 * 2.0):
        super(WW_wrapper, self).__init__()
        self.dt = dt
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)  # Bias is not allowed since it will change the accuracy of the model
        self.WW = WongWangDecisionBP(dt=dt)
        self.histogram = histogram
        self.dist = True

    def forward(self, model_name, logits, directions):
        x = logits * self.w
        decision_times = self.WW(x)
        decision_times = decision_times * (2 * directions - 1).squeeze(1)
        if self.dist:
            decisions_dist = self.histogram(decision_times)
            return decisions_dist
        else:
            return decision_times


def RT_fitting(model_name, logits_path, rt_fitter, histogram, human_rt_dist_train, human_rt_dist_test, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone_logits_list, direction_list = torch.load(f'ckpt/{model_name}/backbone_outputs_{logits_path}.pth')
    backbone_logits_list = backbone_logits_list[::-1]
    direction_list = direction_list[::-1]
    human_rt_dist_train = human_rt_dist_train.to(device)
    model = WW_wrapper(histogram.to(device)).to(device)

    model_parameters = {name: [] for name, p in model.named_parameters() if p.requires_grad}
    parameters_with_grad = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    fitted_model_rt_train = [[] for _ in range(len(config['coherence_list']))]
    loss_curve = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.3)

    for epoch in tqdm(range(config['num_epochs'])):

        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        indices = torch.randperm(config['dataset_size'], device=device)[:config['BATCH_SIZE']]

        for i_coh, coh in enumerate(config['coherence_list']):
            decisions_dist = model(model_name, backbone_logits_list[i_coh][indices], direction_list[i_coh][indices])
            loss = criterion(decisions_dist, human_rt_dist_train[i_coh])
            total_loss += loss

            if epoch % 100 == 99:
                fitted_model_rt_train[i_coh].append(decisions_dist.clone().detach().cpu().numpy())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)

        optimizer.step()
        scheduler.step()

        loss_curve.append(total_loss.item())
        for name, p in parameters_with_grad:
            model_parameters[name].append(p.clone().detach().cpu())

    fitted_model_rt_train = torch.tensor(fitted_model_rt_train)
    for name, p in parameters_with_grad:
        model_parameters[name] = torch.tensor(model_parameters[name])

    model.dist = False
    model.eval()
    fitted_model_rt = [None for _ in range(len(config['coherence_list']))]
    for i_coh, coh in enumerate(config['coherence_list']):
        indices = torch.randperm(config['dataset_size'], device=device)[:config['BATCH_SIZE']]
        decisions_time = model(model_name, backbone_logits_list[i_coh][indices], direction_list[i_coh][indices])
        fitted_model_rt[i_coh] = decisions_time.squeeze().clone().detach().cpu()


    os.makedirs(f'ckpt/{model_name}/{rt_fitter}', exist_ok=True)
    state = {
        'model': model_parameters,
        'loss_curve': loss_curve,
        'fitted_model_rt_train': fitted_model_rt_train,
        'human_rt_dist_train': human_rt_dist_train,
        'fitted_model_rt': fitted_model_rt,
        'human_rt_dist_test': human_rt_dist_test,
    }
    torch.save(state, f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_fit.pth')


    # plotting usage, uncomment to plot the results

    # plt.figure()
    # plt.plot(loss_curve)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss curve')
    # plt.tight_layout()
    # plt.savefig(f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_loss_curve.png')
    #
    # loss_values = []
    # histogram.to('cpu')
    # fig, ax = plt.subplots(4, 2, figsize=(8, 12))
    # axs = ax.ravel()
    # for i, coh in enumerate(config['coherence_list']):
    #     axs[i].plot(histogram.centers.numpy(), human_rt_dist_test[i].cpu().numpy(), label='Human RT')
    #     axs[i].plot(histogram.centers.numpy(), histogram(fitted_model_rt[i]).numpy(), label='Model RT')
    #
    #     loss = criterion(histogram(fitted_model_rt[i]), human_rt_dist_test[i])
    #     loss_values.append(loss.item())
    #
    #     axs[i].set_title(f'Coherence: {coh}')
    #     axs[i].legend(frameon=False)
    #     axs[i].set_xlabel('Reaction time')
    #     axs[i].set_ylabel('Probability')
    #
    # axs[-1].plot(config['coherence_list'], loss_values, 'o-')
    # axs[-1].set_title('Validation loss')
    # axs[-1].set_xlabel('Coherence')
    # axs[-1].set_ylabel('MSE')
    # axs[-1].set_xscale('log')
    # axs[-1].legend([f'Average loss: {np.mean(loss_values):.2e}'], frameon=False)
    #
    # plt.tight_layout()
    # plt.savefig(f'ckpt/{model_name}/{rt_fitter}/{rt_fitter}_fitted_RT.png')


if __name__ == "__main__":
    set_seeds(42)

    histogram = SoftHistogram(500, -2.5, 2.5, 40.0)

    config = {
        'coherence_list': [0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2],
        'BATCH_SIZE': 10000,
        'num_epochs': 5000,
        'lr': 1e-4,
        'dataset_size': 100000
    }
    human_rt_dist_train, human_rt_dist_test = load_human_rt_dataset('../dataset/human_rt_dataset.pkl', config['coherence_list'], histogram)

    RT_fitting('C3D_BN_norm', 'epoch_100', 'WW', histogram, human_rt_dist_train, human_rt_dist_test, config=config)
