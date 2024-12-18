# Please contact Ivan Felipe Rodriguez (ivan_felipe_rodriguez@brown.edu) for any questions over the code.

import os
import sys
import datetime
import json
import math
import random
from pathlib import Path
import urllib.request
from glob import glob

import numpy as np
import pandas as pd

# Do not truncate the contents of cells and display all rows and columns
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from PIL import Image, ImageOps

import timm

from torchtnt.utils import get_module_summary
from torcheval.metrics import MulticlassAccuracy

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import h5py
import multiprocessing

import argparse

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

def tensor_to_pil(tensor, *norm_stats):
    """
    Convert a PyTorch tensor to a PIL Image with optional denormalization.

    Args:
        tensor (torch.Tensor): Input tensor.
        norm_stats (tuple): Optional normalization statistics (mean, std).

    Returns:
        PIL.Image: Output image.
    """
    if norm_stats:
        mean = torch.tensor(norm_stats[0]).view(3, 1, 1)
        std = torch.tensor(norm_stats[1]).view(3, 1, 1)
        tensor = tensor * std + mean
    tensor = tensor.squeeze(0)
    image = TF.to_pil_image(tensor)
    return image

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

def resize_img(img, target_sz, divisor=1):
    """
    Resize the image to the target size.

    Args:
        img (PIL.Image): Input image.
        target_sz (int): Target size for the shortest side.
        divisor (int): Divisor to make the dimensions divisible.

    Returns:
        PIL.Image: Resized image.
    """
    w, h = img.size
    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    return img

class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for handling images.

    Attributes:
        img_paths (list): List of image file paths.
        class_to_idx (dict): Dictionary mapping class names to class indices.
        transforms (callable, optional): Transformations to be applied to the images.
    """

    def __init__(self, img_paths, class_to_idx, transforms=None):
        """
        Initializes the ImageDataset with image paths and transformations.

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
            img_path (Path): Image path.

        Returns:
            tuple: A tuple containing the loaded image and its corresponding label.
        """
        # Load the image from the file path
        image = Image.open(img_path).convert('RGB')

        return image, self._class_to_idx[img_path.parent.name]

def run_epoch(model, dataloader, optimizer, metric, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Run a single training or validation epoch.

    Args:
        model (nn.Module): The model to train or evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        optimizer (Optimizer): Optimizer for training.
        metric (Metric): Metric to evaluate.
        lr_scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run on.
        scaler (GradScaler): Gradient scaler for mixed precision.
        epoch_id (int): Epoch number.
        is_training (bool): Whether it's a training epoch.

    Returns:
        float: Average loss for the epoch.
    """
    # Set model to training or evaluation mode
    model.train() if is_training else model.eval()

    # Reset the performance metric
    metric.reset()
    # Initialize the average loss for the current epoch
    epoch_loss = 0
    # Initialize progress bar with total number of batches in the dataloader
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

    # Iterate over data batches
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Enables gradient calculation if 'is_training' is True
        with torch.set_grad_enabled(is_training):
            # Automatic Mixed Precision (AMP) context manager for improved performance
            with autocast(device_type=device.type):
                outputs = model(inputs)  # Forward pass
                
                loss = torch.nn.functional.cross_entropy(outputs, targets)  # Compute loss

        # Update the performance metric
        metric.update(outputs.detach().cpu(), targets.detach().cpu())

        # If in training mode
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            optimizer.zero_grad()

        loss_item = loss.item()
        epoch_loss += loss_item
        # Update progress bar
        progress_bar.set_postfix(accuracy=metric.compute().item(),
                                 loss=loss_item,
                                 avg_loss=epoch_loss/(batch_id+1),
                                 lr=lr_scheduler.get_last_lr()[0] if is_training else "")
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    progress_bar.close()
    return epoch_loss / (batch_id + 1)

def train_loop(model, train_dataloader, valid_dataloader, optimizer, metric, lr_scheduler, device, epochs, checkpoint_path, use_scaler=False):
    """
    Main training loop.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training set.
        valid_dataloader (DataLoader): DataLoader for the validation set.
        optimizer (Optimizer): Optimizer for training.
        metric (Metric): Metric to evaluate.
        lr_scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run on.
        epochs (int): Number of epochs to train.
        checkpoint_path (Path): Path to save the best model checkpoint.
        use_scaler (bool): Whether to use gradient scaler for mixed precision.
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')

    # Iterate over each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run training epoch and compute training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, metric, lr_scheduler, device, scaler, epoch, is_training=True)
        
        # Run validation epoch and compute validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, metric, None, device, scaler, epoch, is_training=False)

        # If current validation loss is lower than the best one so far, save model and update best loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            metric_value = metric.compute().item()
            torch.save(model.state_dict(), checkpoint_path)
            try:
                model_name = model.name
            except:
                model_name = model.module.name
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'metric_value': metric_value,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model_name
            }

            # Save best_loss and metric_value in a JSON file
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()

def main():
    """
    Main function to run the training script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train an image classifier with PyTorch and timm.')
    parser.add_argument('--dataset-path', type=str, default='kar_dataset', help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save outputs.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--h5-file', type=str, required=True, help='Path to the HDF5 dataset file.')
    args = parser.parse_args()
    
    # Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
    seed = args.seed
    set_seed(seed)
    
    device = get_torch_device()
    dtype = torch.float32
    print(f'Device: {device}, Data Type: {dtype}')
    
    # The name for the project
    project_name = f"pytorch-timm-image-classifier"
    
    # The path for the project folder
    project_dir = Path(args.output_dir) / project_name
    
    # Create the project directory if it does not already exist
    project_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project Directory: {project_dir}")
    
    # Loading and exploring the dataset
    file_path = args.h5_file  # HDF5 dataset file
    # Read H5 file
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
    
    # Create directories and save images
    dataset_path = args.dataset_path
    
    # Create a directory to store the images
    os.makedirs(f'{dataset_path}', exist_ok=True)
    # create training and validation directories
    os.makedirs(f'{dataset_path}/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/val', exist_ok=True)
    
    # create a folder for each class in the training and validation directories. Class is in obj_data
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
    
    # Selecting a model
    base_model = args.model
    model = timm.create_model(base_model, pretrained=True, num_classes=len(classes))
    
    # Set the device and data type for the model
    model = model.to(device=device, dtype=dtype)
    
    # Add attributes to store the device and model name for later reference
    model.device = device
    model.name = f'{base_model}'
    
    # Retrieve normalization statistics (mean and std) specific to the pretrained model
    model_cfg = model.default_cfg
    mean, std = model_cfg['mean'], model_cfg['std']
    norm_stats = (mean, std)
    print(f"Normalization stats: mean={mean}, std={std}")
    
    # Preparing the data
    train_sz = 224
    
    # Data augmentation
    # Set the fill color for padding images
    fill = (0,0,0)
    
    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=train_sz)
    
    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=fill)
    
    # Create a TrivialAugmentWide object
    trivial_aug = transforms.TrivialAugmentWide(fill=fill)
    
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
    
    # Convert class names to strings
    class_names = [str(c) for c in classes]
    class_names.sort()
    
    # Initialize datasets
    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    # Instantiate the dataset using the defined transformations
    train_dataset = ImageDataset(train_paths, class_to_idx, train_tfms)
    valid_dataset = ImageDataset(val_paths, class_to_idx, valid_tfms)
    
    # Print the number of samples in the training and validation datasets
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(valid_dataset)}')
    
    # Training batch size
    bs = args.batch_size
    
    # Initialize DataLoaders
    # Set the number of worker processes for loading data. This should be the number of CPUs available.
    num_workers = min(multiprocessing.cpu_count(), 8)
    
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
    
    # Set the Model Checkpoint Path
    # Generate timestamp for the training session
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a directory to store the checkpoints
    checkpoint_dir = Path(project_dir/f"{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # The model checkpoint path
    checkpoint_path = checkpoint_dir/f"{model.name}.pth"
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Saving the class labels
    dataset_name = Path(dataset_path).stem
    class_labels = {"classes": list(class_names)}
    
    # Set file path
    class_labels_path = checkpoint_dir/f"{dataset_name}-classes.json"
    
    # Save class labels in JSON format
    with open(class_labels_path, "w") as write_file:
        json.dump(class_labels, write_file)
    
    print(f"Class labels saved to: {class_labels_path}")
    
    # Configure the Training Parameters
    # Learning rate for the model
    lr = args.learning_rate
    # Number of training epochs
    epochs = args.epochs
    # AdamW optimizer; includes weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
    # Learning rate scheduler; adjusts the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=lr,
                                                       total_steps=epochs*len(train_dataloader))
    
    # Performance metric: Multiclass Accuracy
    metric = MulticlassAccuracy()
    
    # Train the model
    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               metric=metric,
               lr_scheduler=lr_scheduler,
               device=torch.device(device),
               epochs=epochs,
               checkpoint_path=checkpoint_path,
               use_scaler=True
              )
    
    # Making predictions with the model
    outputs = []
    # Choose an item from the validation set
    
    model.eval()
    for test_file in val_paths:
        # Open the test file
        test_img = Image.open(test_file).convert('RGB')
    
        # Set the minimum input dimension for inference
        input_img = resize_img(test_img, target_sz=train_sz, divisor=1)
    
        # Convert the image to a normalized tensor and move it to the device
        img_tensor = pil_to_tensor(input_img, *norm_stats).to(device=device)
    
        # Make a prediction with the model
        with torch.no_grad():
            pred = model(img_tensor)
        
        # Scale the model predictions to add up to 1
        pred_scores = torch.softmax(pred, dim=1)
        
        # Get the highest confidence score
        confidence_score = pred_scores.max()
        
        # Get the class index with the highest confidence score and convert it to the class name
        pred_class = class_names[torch.argmax(pred_scores)]
        
        outputs.append([str(test_file), pred_class, confidence_score.item()])
        
    outputs_df = pd.DataFrame(outputs, columns=['image_path', 'pred_class', 'confidence_score'])
    
    # Save predictions
    outputs_df.to_csv(f'{checkpoint_dir}/validation_predictions.csv', index=False)
    print(f"Predictions saved to {checkpoint_dir}/validation_predictions.csv")

if __name__ == '__main__':
    main()
