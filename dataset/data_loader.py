import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.RDM_generator import RDM_generator
from PIL import Image
import pandas as pd
from torchvision import transforms
import re

class RDMDataset(Dataset):
    def __init__(self, coherence, func, size, img_size=126, dot_size=3, num_frames=150, speed=5, groups=3, nDot=6):
        self.coherence = coherence
        self.directions = [0, 180]
        self.func = func
        self.size = size
        self.dot_size = dot_size
        self.img_size = img_size
        self.speed = speed
        self.num_frames = num_frames
        self.groups = groups
        self.nDot = nDot

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of bounds of dataset.")
        
        direction = self.directions[idx % 2]
        RDM = self.func(direction=direction,
                        coherence=self.coherence,
                        size=self.img_size,
                        dotSize=self.dot_size,
                        frames=self.num_frames,
                        speed=self.speed,
                        groups=self.groups,
                        nDot=self.nDot)

        RDM = torch.tensor(RDM, dtype=torch.float32)
        direction = torch.tensor(direction // 180, dtype=torch.float32)

        RDM = RDM.unsqueeze(1).repeat(1, 3, 1, 1)
        direction = direction.unsqueeze(0)

        return RDM, direction


class KarDataset_logits(Dataset):
    def __init__(self, df, rt_df, model_name):
        super(Dataset, self).__init__()
        # Filter out rows with empty 'ost' values
        df = df[df['ost'].notna()]

        self._img_paths = df['image_path'].tolist()
        self._classes = df['gt'].tolist()
        self._ost = df['ost'].tolist()
        self._logits = df['pred'].tolist()
        self._out = df['out'].tolist()
        self._model_name = model_name

        # print the acc
        correct = 0.0
        for i in range(len(self._classes)):
            correct += self._classes[i] == self._out[i]
        acc = (correct)/len(self._classes)
        print(f"Accuracy: {acc}")

        if model_name.startswith('vgg'):
            self._mapping = {1: 0, 10:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
        elif model_name.startswith('cornet'):
            self._mapping = {1: 0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

        # Apply the mapping to the classes list
        mapped_classes = [self._mapping[x] for x in self._classes]
        self._classes = mapped_classes

        self._rt = []
        for i in range(len(self._ost)):
            img_name = float(self._img_paths[i].split('/')[-1][:-4])
            diff = np.abs(rt_df['i1'].values - img_name)
            rt = rt_df.iloc[np.argmin(diff)]['rt_median']
            self._rt.append(rt)

        for i in range(len(self._logits)):
            fixed_str = re.sub(r'\s+', ', ', self._logits[i].replace('\n', '').strip()).replace('[,', '[')
            self._logits[i] = torch.tensor(eval(fixed_str)).squeeze()

    def __len__(self):
        return len(self._ost)

    def __getitem__(self, index):
        label = torch.tensor(self._classes[index], dtype=torch.long)
        rt = torch.tensor(self._rt[index], dtype=torch.float32)
        logit = self._logits[index]
        return label, rt.squeeze(), logit


class KarDataset(Dataset):
    def __init__(self, data, rt_df, model_name):
        self._img_paths = data['image_path']
        self._classes = data['gt']
        self._ost = data['ost']
        self._out = data['out']
        self._pred = data['pred']
        self._time_logits = data['time steps']
        self._hidden_states = data['hidden_states']
        self.model_name = model_name

        # filter out the rows where the ost is nan
        self._img_paths = [self._img_paths[i] for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._classes = [self._classes[i] for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._out = [eval(self._out[i]) for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._pred = [self._pred[i] for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._time_logits = [self._time_logits[i] for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._hidden_states = [torch.concatenate(self._hidden_states[i], dim=0) for i in range(len(self._ost)) if not np.isnan(self._ost[i])]
        self._ost = [self._ost[i] for i in range(len(self._ost)) if not np.isnan(self._ost[i])]

        # print the acc
        correct = 0.0
        for i in range(len(self._classes)):
            correct += self._classes[i] == self._out[i]
        acc = (correct) / len(self._classes)
        print(f"Accuracy: {acc}")

        if model_name.startswith('vgg'):
            self._mapping = {1: 0, 10:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
        elif model_name.startswith('cornet'):
            self._mapping = {1: 0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

        # Apply the mapping to the classes list
        mapped_classes = [self._mapping[x] for x in self._classes]
        self._classes = mapped_classes

        self.rt = []
        for i in range(len(self._ost)):
            if '\\' in str(self._img_paths[i]):
                img_name = float(str(self._img_paths[i]).split('\\')[-1][:-4])
            else:
                img_name = float(str(self._img_paths[i]).split('/')[-1][:-4])
            diff = np.abs(rt_df['i1'].values - img_name)
            rt = rt_df.iloc[np.argmin(diff)]['rt_median']
            self.rt.append(rt)

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        label = torch.tensor(self._classes[idx], dtype=torch.long)
        rt = torch.tensor(self.rt[idx], dtype=torch.float32)
        time_logits = torch.tensor(self._time_logits[idx], dtype=torch.float32).squeeze()
        hidden_states = self._hidden_states[idx]

        return label, rt.squeeze(), time_logits, hidden_states


if __name__ == '__main__':

    rt_df = pd.read_csv('../Kar/Kar_dataset/human_rt_data.csv')
    hidden_states = np.load('../Kar/Kar_dataset/hidden_states.npy', allow_pickle=True).item()
    train_hidden, test_hidden = hidden_states['train'], hidden_states['val']

    train_dict = {}
    test_dict = {}

    for key in train_hidden[0].keys():
        train_dict[key] = [d[key] for d in train_hidden]
        test_dict[key] = [d[key] for d in test_hidden]

    train_dataset = KarDataset(train_dict, rt_df, 'cornet')
    valid_dataset = KarDataset(test_dict, rt_df, 'cornet')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    rt_df = pd.read_csv('../Kar/Kar_dataset/human_rt_data.csv')
    train_df = pd.read_csv('../Kar/Kar_dataset/train_logits_vgg1.csv')
    val_df = pd.read_csv('../Kar/Kar_dataset/test_logits_vgg1.csv')

    train_dataset = KarDataset_logits(train_df, rt_df, 'vgg')
    valid_dataset = KarDataset_logits(val_df, rt_df, 'vgg')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)



