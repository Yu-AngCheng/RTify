# Please contact Ivan Felipe Rodriguez (ivan_felipe_rodriguez@brown.edu) for any questions over the code.

import os
import sys
import datetime
import json
import math
import random
from pathlib import Path
import urllib.request
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF

from PIL import Image, ImageOps

import timm

from tqdm.auto import tqdm

from scipy.ndimage import gaussian_filter

from sklearn.model_selection import train_test_split


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_torch_device():
    """
    Get the available torch device.

    Returns:
        str: 'cuda' if CUDA is available, else 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def pil_to_tensor(image, *norm_stats):
    """
    Convert a PIL Image to a PyTorch tensor with optional normalization.

    Args:
        image (PIL.Image): Input image.
        norm_stats (tuple): Optional normalization statistics (mean, std).

    Returns:
        torch.Tensor: Image tensor.
    """
    transform_list = [transforms.ToTensor()]
    if norm_stats:
        transform_list.append(transforms.Normalize(*norm_stats))
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)


def denorm_img_tensor(tensor, mean, std):
    """
    Denormalize an image tensor.

    Args:
        tensor (torch.Tensor): Normalized image tensor.
        mean (list): Mean used for normalization.
        std (list): Std used for normalization.

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def get_img_files(folder):
    """
    Get all image files in a folder with common image extensions.

    Args:
        folder (Path): Path to the folder.

    Returns:
        list: List of image file paths.
    """
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*.{ext}'))
        files.extend(folder.glob(f'*.{ext.upper()}'))
    return files


class ResizeMax:
    """
    Resize the image so that the longest side matches max_sz.

    Attributes:
        max_sz (int): Maximum size for the longest side.
    """
    def __init__(self, max_sz):
        self.max_sz = max_sz

    def __call__(self, img):
        """
        Apply the resizing to the image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Resized image.
        """
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > self.max_sz:
            scale = self.max_sz / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.ANTIALIAS)
        return img


class PadSquare:
    """
    Pad the image to make it square.

    Attributes:
        shift (bool): Whether to randomly shift the image within the padding.
        fill (tuple): Fill color for padding.
    """
    def __init__(self, shift=False, fill=(0, 0, 0)):
        self.shift = shift
        self.fill = fill

    def __call__(self, img):
        """
        Apply the padding to the image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Padded image.
        """
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        if self.shift:
            h_shift = random.randint(-hp, hp)
            v_shift = random.randint(-vp, vp)
        else:
            h_shift = 0
            v_shift = 0
        padding = (hp - h_shift, vp - v_shift, hp + h_shift, vp + v_shift)
        img = ImageOps.expand(img, padding, fill=self.fill)
        return img


def show(img, p=False, smooth=False, inverse_channels=False, **kwargs):
    """ Display torch/tf tensor """
    img = np.array(img, dtype=np.float32)

    # check if channel first
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
    # check if cmap
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    if img.shape[-1] == 3 and inverse_channels:
        img = img[..., ::-1]
    # normalize
    if img.max() > 1 or img.min() < 0:
        img -= img.min()
        img /= img.max()
    # check if clip percentile
    if p is not False:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100 - p))

    if smooth and len(img.shape) == 2:
        img = gaussian_filter(img, smooth)

    plt.imshow(img, **kwargs)
    plt.axis('off')
    plt.grid(None)
def generate_predictions_with_hidden_states(model, train_paths, val_paths, class_names, i1_data, ost_data, obj_data, norm_stats, device, train_sz):
    """
    Generate predictions and collect hidden states for both training and validation datasets.

    Args:
        model (nn.Module): The trained model.
        train_paths (list): List of training image paths.
        val_paths (list): List of validation image paths.
        class_names (list): List of class names.
        i1_data (list): List of image identifiers.
        ost_data (list): List of ost data corresponding to images.
        obj_data (list): List of object class data corresponding to images.
        norm_stats (tuple): Normalization statistics (mean, std).
        device (torch.device): Device to run on.
        train_sz (int): Target image size.

    Returns:
        dict: Dictionary containing predictions and hidden states.
    """
    counter = 1
    final_dict = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            allpaths = train_paths
        else:
            allpaths = val_paths
        outputs = []
        for test_file in allpaths:
            if counter % 100 == 1:
                print('progress', counter)
            # Open the test file
            i1 = float(str(test_file).split('/')[-1].split('.jpg')[0])
            index = list(i1_data).index(i1)
            ost = ost_data[index]
            cl = obj_data[index]
            test_img = Image.open(test_file).convert('RGB')

            # Set the minimum input dimension for inference
            input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

            # Convert the image to a normalized tensor and move it to the device
            img_tensor = pil_to_tensor(input_img, *norm_stats).to(device=device)

            # Make a prediction with the model
            with torch.no_grad():
                pred, outs_ts, out_hd = model(img_tensor, inference=True)

            counter += 1
            out = class_names[np.argmax(pred.cpu().numpy())]
            print(f'Predicted class: {out}')
            outputs.append([str(test_file), cl, ost, out, pred.cpu().numpy(), outs_ts.cpu().numpy(), out_hd])

        columns = ['image_path', 'gt', 'ost', 'out', 'pred', 'time_steps', 'hidden_states']
        final_dict[split] = [{k: v for k, v in zip(columns, outputs[i])} for i in range(len(outputs))]

    np.save('predictions_with_hidden_states.pkl', final_dict, allow_pickle=True)
    print("Predictions with hidden states saved to 'predictions_with_hidden_states.pkl'")
    return final_dict

def main():
    # Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
    seed = 1234
    set_seed(seed)

    device = get_torch_device()
    dtype = torch.float32
    print(f'Device: {device}, Data Type: {dtype}')

    # The name for the project
    project_name = f"pytorch-timm-image-classifier"

    # The path for the project folder
    project_dir = Path(f"./{project_name}/")

    # Create the project directory if it does not already exist
    project_dir.mkdir(parents=True, exist_ok=True)

    # Define path to store datasets
    dataset_dir = Path("./")
    # Create the dataset directory if it does not exist
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Define path to store archive files
    archive_dir = dataset_dir / '../Archive'
    # Create the archive directory if it does not exist
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project Directory: {project_dir}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Archive Directory: {archive_dir}")

    # Loading and exploring the dataset
    file_path = '../../data/dataset (1).h5'  # Replace with your dataset path
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Access the datasets within the file
        i1_data = file['i1'][:][0]
        images_data = file['images'][:]
        obj_data = file['obj'][:][0]
        ost_data = file['ost'][:][0]

    print(f"Data lengths: i1_data={len(i1_data)}, images_data={len(images_data)}, obj_data={len(obj_data)}, ost_data={len(ost_data)}")

    # Split the data into training and validation sets
    train_images, val_images, train_i1, val_i1, train_obj, val_obj, train_ost, val_ost = train_test_split(
        images_data, i1_data, obj_data, ost_data, test_size=0.2, random_state=seed)

    dataset_path = 'kar_dataset_50'

    # Create a directory to store the images
    os.makedirs(f'{dataset_path}', exist_ok=True)
    # Create training and validation directories
    os.makedirs(f'{dataset_path}/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/val', exist_ok=True)

    # Create a folder for each class in the training and validation directories. Class is in obj_data
    classes = np.unique(obj_data)
    for class_name in classes:
        os.makedirs(f'{dataset_path}/train/{class_name}', exist_ok=True)
        os.makedirs(f'{dataset_path}/val/{class_name}', exist_ok=True)

    # Iterate over the images and save them to the directory for training
    data = []
    for i, (img, i1, obj, ost) in enumerate(zip(train_images, train_i1, train_obj, train_ost)):
        img = np.uint8(img.transpose(1, 2, 0))
        img = Image.fromarray(img)
        img.save(f'{dataset_path}/train/{obj}/{i1}.jpg')
        data.append([f'{dataset_path}/train/{obj}/{i1}.jpg', obj, ost])
    data_df = pd.DataFrame(data, columns=['image_path', 'obj', 'ost'])
    data_df.to_csv(f'{dataset_path}/train.csv', index=False)

    # Iterate over the images and save them to the directory for validation
    data = []
    for i, (img, i1, obj, ost) in enumerate(zip(val_images, val_i1, val_obj, val_ost)):
        img = np.uint8(img.transpose(1, 2, 0))
        img = Image.fromarray(img)
        img.save(f'{dataset_path}/val/{obj}/{i1}.jpg')
        data.append([f'{dataset_path}/val/{obj}/{i1}.jpg', obj, ost])
    data_df = pd.DataFrame(data, columns=['image_path', 'obj', 'ost'])
    data_df.to_csv(f'{dataset_path}/val.csv', index=False)

    # Get image folders
    train_dataset_path = Path(f'{dataset_path}/train')
    train_img_folder_paths = [folder for folder in train_dataset_path.iterdir() if folder.is_dir()]

    # Get image file paths
    train_class_file_paths = [get_img_files(folder) for folder in train_img_folder_paths]

    # Get all image files in the 'img_dir' directory
    train_paths = [
        file
        for folder in train_class_file_paths  # Iterate through each image folder
        for file in folder  # Get a list of image files in each image folder
    ]

    val_dataset_path = Path(f'{dataset_path}/val')
    val_img_folder_paths = [folder for folder in val_dataset_path.iterdir() if folder.is_dir()]

    val_class_file_paths = [get_img_files(folder) for folder in val_img_folder_paths]

    val_paths = [
        file
        for folder in val_class_file_paths  # Iterate through each image folder
        for file in folder  # Get a list of image files in each image folder
    ]

    # Print the number of images in the training and validation sets
    print(f"Training Samples: {len(train_paths)}")
    print(f"Validation Samples: {len(val_paths)}")

    # Selecting a Model
    # Note: Replace 'CORnet_S_t' with the actual model definition or import
    from models.cornet_s import CORnet_S_t  # Ensure this module is available
    model = CORnet_S_t(times=20)

    # Set the device and data type for the model
    model = model.to(device=device, dtype=dtype)

    # Add attributes to store the device and model name for later reference
    model.device = device

    base_model = 'cornet-z'
    version = "imagenet1k"
    model.name = f'{base_model}.{version}'

    # Preparing the Data
    train_sz = 224

    # Data Augmentation
    # Set the fill color for padding images
    fill = (0, 0, 0)

    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=train_sz)

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=fill)

    # Create a TrivialAugmentWide object
    trivial_aug = transforms.TrivialAugmentWide(fill=fill)

    # Retrieve normalization statistics (mean and std) specific to the pretrained model
    # Here, we use standard ImageNet normalization stats
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm_stats = (mean, std)

    # Image Transforms
    # Compose transforms to resize and pad input images
    resize_pad_tfm = transforms.Compose([
        resize_max,
        pad_square,
        transforms.Resize([train_sz] * 2, antialias=True)
    ])

    # Compose transforms to sanitize bounding boxes and normalize input data
    final_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])

    # Define the transformations for training and validation datasets
    # Note: Data augmentation is performed only on the training dataset
    train_tfms = transforms.Compose([
        trivial_aug,
        resize_pad_tfm,
        final_tfms
    ])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])

    # Convert class names to strings and sort
    class_names = [str(c) for c in classes]
    class_names.sort()

    # Initialize Datasets
    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Training Dataset Class
    class ImageDataset(Dataset):
        """
        A PyTorch Dataset class for handling images.

        This class extends PyTorch's Dataset and is designed to work with image data.
        It supports loading images, and applying transformations.

        Attributes:
            img_paths (list): List of image file paths.
            class_to_idx (dict): Dictionary mapping class names to class indices.
            transforms (callable, optional): Transformations to be applied to the images.
        """

        def __init__(self, img_paths, class_to_idx, transforms=None):
            """
            Initializes the ImageDataset with image keys and other relevant information.

            Args:
                img_paths (list): List of image file paths.
                class_to_idx (dict): Dictionary mapping class names to class indices.
                transforms (callable, optional): Transformations to be applied to the images.
            """
            super(Dataset, self).__init__()

            self._img_paths = img_paths
            self._class_to_idx = class_to_idx
            self._transforms = transforms

        def __len__(self):
            """
            Returns the number of items in the dataset.

            Returns:
                int: Number of items in the dataset.
            """
            return len(self._img_paths)

        def __getitem__(self, index):
            """
            Retrieves an item from the dataset at the specified index.

            Args:
                index (int): Index of the item to retrieve.

            Returns:
                tuple: A tuple containing the image and its corresponding label.
            """
            img_path = self._img_paths[index]
            image, label = self._load_image(img_path)

            # Applying transformations if specified
            if self._transforms:
                image = self._transforms(image)

            return image, label

        def _load_image(self, img_path):
            """
            Loads an image from the provided image path.

            Args:
                img_path (string): Image path.

            Returns:
                tuple: A tuple containing the loaded image and its corresponding target data.
            """
            # Load the image from the file path
            image = Image.open(img_path).convert('RGB')

            return image, self._class_to_idx[img_path.parent.name]

    # Instantiate the dataset using the defined transformations
    train_dataset = ImageDataset(train_paths, class_to_idx, train_tfms)
    valid_dataset = ImageDataset(val_paths, class_to_idx, valid_tfms)

    # Print the number of samples in the training and validation datasets
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    # Initialize DataLoaders
    bs = 32  # Training Batch Size
    num_workers = 1

    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': bs,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,  # Keep workers alive after dataset is consumed
        'pin_memory': 'cuda' in device,  # Copy tensors into CUDA pinned memory before returning them
        'pin_memory_device': device if 'cuda' in device else '',  # Device for pin_memory
    }

    # Create DataLoader for training data. Data is shuffled for every epoch.
    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

    # Create DataLoader for validation data. Shuffling is not necessary for validation data.
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    print(f'Number of batches in train DataLoader: {len(train_dataloader)}')
    print(f'Number of batches in validation DataLoader: {len(valid_dataloader)}')

    # Generate Predictions with Hidden States
    final_dict = generate_predictions_with_hidden_states(
        model=model,
        train_paths=train_paths,
        val_paths=val_paths,
        class_names=class_names,
        i1_data=i1_data,
        ost_data=ost_data,
        obj_data=obj_data,
        norm_stats=norm_stats,
        device=device,
        train_sz=train_sz
    )
    # Making Predictions with the Model
    # Define necessary classes for saliency map computation
    import torch
    from torch.autograd import Function

    class DiffDecision(Function):
        @staticmethod
        def forward(ctx, trajectory, dsdt_trajectory):
            mask = trajectory > 0
            decision_time = mask.float().argmax(dim=1).float()
            decision_time[mask.sum(dim=1) == 0] = torch.tensor(trajectory.shape[1] - 1, dtype=torch.float32)
            ctx.save_for_backward(dsdt_trajectory, decision_time)
            return decision_time

        @staticmethod
        def backward(ctx, grad_output):
            dsdt_trajectory, decision_times = ctx.saved_tensors
            grads = torch.zeros_like(dsdt_trajectory)
            batch_indices = torch.arange(decision_times.size(0)).to(decision_times.device)
            grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[
                batch_indices, decision_times.long()] + 1e-6)
            grads = grads * grad_output.unsqueeze(1).expand_as(grads)
            return grads, None

    class DiffDecisionMultiClass(Function):
        @staticmethod
        def forward(ctx, trajectory, dsdt_trajectory):
            mask = trajectory > 0
            decision_times = mask.float().argmax(dim=1).float()
            decision_times[mask.sum(dim=1) == 0] = torch.tensor(trajectory.size(1) - 1, dtype=torch.float32, device=trajectory.device)
            ctx.save_for_backward(dsdt_trajectory, decision_times)
            return decision_times

        @staticmethod
        def backward(ctx, grad_output):
            dsdt_trajectory, decision_times = ctx.saved_tensors
            grads = torch.zeros_like(dsdt_trajectory)
            decision_indices = decision_times.long()
            batch_indices, class_indices = torch.meshgrid(
                torch.arange(decision_times.size(0), device=decision_times.device),
                torch.arange(decision_times.size(1), device=decision_times.device), indexing='ij')
            grads[batch_indices, decision_indices[batch_indices, class_indices], class_indices] = -1.0 / (dsdt_trajectory[
                                    batch_indices, decision_indices[batch_indices, class_indices], class_indices] + 1e-6)
            grads = grads * grad_output.unsqueeze(1).expand_as(grads)
            return grads, None

    class Flatten(nn.Module):
        """
        Helper module for flattening input tensor to 1-D for the use in Linear modules
        """
        def forward(self, x):
            return x.view(x.size(0), -1)

    class Identity(nn.Module):
        """
        Helper module that stores the current tensor. Useful for accessing by name
        """
        def forward(self, x):
            return x

    class cornet_wrapper(nn.Module):
        def __init__(self, time_steps=20, sigma=2.0):
            super(cornet_wrapper, self).__init__()
            self.time_steps = time_steps
            self.fc2 = nn.Linear(time_steps, 1, bias=False)
            self.decoder = nn.Sequential(nn.ModuleDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(512, len(class_names))),
                ('output', Identity())
            ]))
            self.w = nn.Parameter(torch.ones((1, time_steps)) * 0.1)
            self.b = nn.Parameter(torch.zeros((1, time_steps)))
            self.linear = nn.Sequential(nn.ModuleDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear1', nn.Linear(512, 64)),
                ('ReLU', nn.ReLU()),
                ('linear2', nn.Linear(64, 64)),
                ('ReLU', nn.ReLU()),
                ('linear3', nn.Linear(64, len(class_names))),
                ('output', Identity())
            ]))
            self.threshold = torch.nn.Parameter(torch.tensor(1.00))  # threshold for the decision
            self.sigma = sigma

        def forward(self, hidden_state):
            s_traj = (self.linear(hidden_state.view(-1, 512, 7, 7)).view(-1, self.time_steps, len(class_names)).permute(0, 2, 1) * self.w + self.b).permute(0, 2, 1)
            s_accumulated = torch.cumsum(s_traj, dim=1)
            dsdt_trajectory = torch.diff(s_accumulated, dim=1)
            dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)
            decision_time = DiffDecisionMultiClass.apply(s_accumulated - self.threshold, dsdt_trajectory)
            decision_time_min = decision_time.min(dim=1).values
            soft_index = torch.sigmoid((decision_time_min.unsqueeze(1) - torch.arange(self.time_steps, device=decision_time_min.device) + 0.5) * self.sigma)
            logit_trajectory = self.decoder(hidden_state.view(-1, 512, 7, 7)).view(-1, self.time_steps, len(class_names)).permute(0, 2, 1) * self.fc2.weight
            decision_logits = (logit_trajectory * soft_index.unsqueeze(1)).sum(dim=2)
            return decision_logits, decision_time_min
    
    # Load the models (Ensure checkpoints are available at specified paths)
    model_dec_rt = cornet_wrapper()
    ckpt = torch.load('modeltemp_ckpt.pth')  # Ensure this file is available
    model_dec_rt.load_state_dict(ckpt['human_fit'])
    model_dec_rt = model_dec_rt.to(device)
    model_dec = cornet_wrapper()
    model_dec.load_state_dict(ckpt['self_penalty'])
    model_dec = model_dec.to(device)

    # Generate Saliency Maps
    outputs = []
    folder_saliency = f'{dataset_path}/saliency_maps/'
    os.makedirs(folder_saliency, exist_ok=True)

    model.eval()
    model_dec.eval()
    model_dec_rt.eval()

    # Loop through the test files
    for test_file in val_paths:
        print(f"Processing {test_file}")
        name = str(test_file).split('/')[-1][:-3] + 'pdf'

        # Open the test file and preprocess
        test_img = Image.open(test_file).convert('RGB')
        input_img = resize_img(test_img, target_sz=train_sz, divisor=1)
        img_tensor = pil_to_tensor(input_img, *norm_stats).to(device=device)

        # Classification saliency map
        img_tensor.requires_grad = True
        pred, outs_ts, out_hd = model(img_tensor, inference=True)
        pred.backward(torch.ones_like(pred))
        gradients_pred = img_tensor.grad.data
        saliency_map = torch.abs(gradients_pred).sum(dim=1).squeeze().detach().cpu().numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        plt.subplot(1, 2, 1)
        show(input_img)
        show(saliency_map, p=1, smooth=2, alpha=0.7, cmap='viridis')
        plt.title('Classification')

        # RTify self-penalized saliency map
        img_tensor.grad.zero_()
        pred, outs_ts, out_hd = model(img_tensor, inference=True)
        out_hd = torch.cat(out_hd).to(device)
        logits_dec, rt_dec = model_dec(out_hd)
        rt_dec.backward(torch.ones_like(rt_dec))
        gradients_dc = img_tensor.grad.data
        saliency_map = torch.abs(gradients_dc).sum(dim=1).squeeze().detach().cpu().numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        plt.subplot(1, 2, 2)
        show(input_img)
        show(saliency_map, p=1, smooth=2, alpha=0.7)
        plt.title('RTify Self-Penalized')
        plt.savefig(f'{folder_saliency}/comp_{name}')
        plt.show()


if __name__ == '__main__':
    main()
