import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset.data_loader import RDMDataset
from dataset.RDM_generator import RDM_generator
from tqdm import tqdm


def train_model(model, criterion, optimizer, num_epochs, model_name, coherence_list=None, lr_scheduler=None, norm=False, curiculum=False):

    if coherence_list is None:
        coherence_list = [99.9, 51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]

    BATCH_SIZE = 50
    stim_size = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_size = 5000
    time_steps = 150

    coherence_acc_list = [[] for _ in range(len(coherence_list))]
    coherence_loss_list = [[] for _ in range(len(coherence_list))]
    epoch_acc_list = []
    epoch_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        total_acc = 0.0

        if curiculum:
            if epoch < 10:
                coherence_list = [99.9]
            else:
                coherence_list = [99.9, 51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]

        for i_conherence, coherence in enumerate(coherence_list):
            # All stimuli are three times smaller
            dataset = RDMDataset(coherence, RDM_generator, size=dataset_size, img_size=stim_size, num_frames=time_steps,
                                 speed=5.0/3, dot_size=1, groups=3, nDot=6)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            running_loss = 0.0
            running_acc = 0.0

            for batch_idx, (sequences, train_directions) in enumerate(dataloader):
                sequences, train_directions = sequences.to(device), train_directions.to(device)

                if norm:
                    sequences /= 255.0

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, train_directions)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * sequences.size(0)
                total_loss += loss.item() * sequences.size(0)

                batch_acc = torch.sum((outputs > 0) == (train_directions > 0)) / BATCH_SIZE
                running_acc += batch_acc.item() * sequences.size(0)
                total_acc += batch_acc.item() * sequences.size(0)

                # print(f"batch {batch_idx} loss: {loss.item():.4f}")
                # print(f"batch {batch_idx} accuracy: {batch_acc:.4f}")

            coherence_acc = running_acc / dataset_size
            coherence_loss = running_loss / dataset_size
            coherence_acc_list[i_conherence].append(coherence_acc)
            coherence_loss_list[i_conherence].append(coherence_loss)

            print(f"Epoch {epoch}/{num_epochs - 1} Coherence {coherence} Loss: {coherence_loss:.4f}")
            print(f"Epoch {epoch}/{num_epochs - 1} Coherence {coherence} Accuracy: {coherence_acc:.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_loss = total_loss / dataset_size / len(coherence_list)
        epoch_acc = total_acc / dataset_size / len(coherence_list)
        epoch_acc_list.append(epoch_acc)
        epoch_loss_list.append(epoch_loss)

        print(f"Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f}")
        print(f'Epoch {epoch}/{num_epochs - 1} Accuracy: {epoch_acc:.4f}')

        os.makedirs(f'ckpt/{model_name}', exist_ok=True)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_acc_list': epoch_acc_list,
            'epoch_loss_list': epoch_loss_list,
            'coherence_acc_list': coherence_acc_list,
            'coherence_loss_list': coherence_loss_list
        }
        torch.save(state, f'ckpt/{model_name}/epoch_{epoch}.pth')

    print("Training complete!")


# precompute the hidden_states/logits
def model_outputs(model, ckpt_path, model_name, BATCH_SIZE, coherence_list=None, norm=False, dynamic=False, dataset_size=100000):

    checkpoint = torch.load(f'ckpt/{model_name}/{ckpt_path}.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if coherence_list is None:
        coherence_list = [51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]

    stim_size = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone_outputs_list = [None for _ in range(len(coherence_list))]
    direction_list = [None for _ in range(len(coherence_list))]

    with torch.no_grad():

        for i_coherence, coherence in enumerate(coherence_list):
            dataset = RDMDataset(coherence, RDM_generator, size=dataset_size, img_size=stim_size, num_frames=150,
                                 speed=5.0 / 3, dot_size=1, groups=3, nDot=6)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for batch_idx, (sequences, train_directions) in tqdm(enumerate(dataloader), total=len(dataloader)):
                sequences, train_directions = sequences.to(device), train_directions.to(device)

                if norm:
                    sequences /= 255.0

                backbone_outputs_temp = model(sequences).to(device)
                backbone_outputs_temp = backbone_outputs_temp.cpu()
                assert torch.isnan(backbone_outputs_temp).any() == False

                if backbone_outputs_list[i_coherence] is None:
                    backbone_outputs_list[i_coherence] = backbone_outputs_temp
                else:
                    backbone_outputs_list[i_coherence] = torch.cat((backbone_outputs_list[i_coherence], backbone_outputs_temp), 0)

                if direction_list[i_coherence] is None:
                    direction_list[i_coherence] = train_directions
                else:
                    direction_list[i_coherence] = torch.cat((direction_list[i_coherence], train_directions), 0)

    # save the precomputed backbone hidden states/logits and directions in one file
    if dynamic is False:
        torch.save((backbone_outputs_list, direction_list), f'ckpt/{model_name}/backbone_outputs_{ckpt_path}.pth')
    else:
        torch.save((backbone_outputs_list, direction_list), f'ckpt/{model_name}/backbone_outputs_{ckpt_path}_dynamic.pth')


if __name__ == '__main__':
    from utils import set_seeds
    from AlexNet_BN_LSTM_backbone import AlexNet_BN_LSTM as AlexNet_BN_LSTM
    from C3D_BN_backbone import C3D_BN
    from AlexNet_BN_LSTM_backbone_2 import AlexNet_BN_LSTM as AlexNet_BN_LSTM2

    set_seeds()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_outputs(C3D_BN().to(device), 'epoch_100', 'C3D_BN_norm', BATCH_SIZE=50, norm=True)

    model = AlexNet_BN_LSTM(time_steps=150).to(device)
    model.eval()
    model.output = True
    model_outputs(model, 'epoch_100', 'AlexNet_BN_LSTM_norm', BATCH_SIZE=512, norm=True, dynamic=True, dataset_size=2048)

    model = AlexNet_BN_LSTM2(time_steps=150).to(device)
    model.eval()
    model.output = True
    model_outputs(model, 'epoch_100', 'AlexNet_BN_LSTM_norm_2', BATCH_SIZE=512, norm=True, dynamic=True, dataset_size=2048)


