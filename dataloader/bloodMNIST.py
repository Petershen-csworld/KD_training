import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple
import os
import medmnist
from medmnist import INFO
import numpy as np

# BloodMNIST mean and std (hypothetical values, replace with actual if available)
BLOODMNIST_MEAN = [0.5]
BLOODMNIST_STD = [0.5]



def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    # Convert targets to a single NumPy array first
    targets = np.stack(targets, axis=0)
    # Convert the NumPy array to a PyTorch tensor
    targets = torch.from_numpy(targets)
    return inputs, targets.squeeze(dim=1)  # Squeeze the targets here


def get_bloodmnist_dataloaders(root: str = "./data",
                               train_transform: transforms.transforms = None,
                               test_transform: transforms.transforms = None,
                               batch_size: int = 128,
                               num_workers: int = 8
                               ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_ds = DataClass(
        split='train', transform=train_transform, download=True, root=root)
    test_ds = DataClass(
        split='test', transform=test_transform, download=True, root=root)
    # val_ds = DataClass(split='val', transform=test_transform,
    #                    download=True, root=root)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=1 if num_workers == 1 else num_workers // 2,
                             pin_memory=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
    #                         num_workers=1 if num_workers == 1 else num_workers // 2, pin_memory=True)

    return train_loader, test_loader


def bloodmnist_crop_flip_transform(train: bool = True) -> transforms:
    transform = []
    if train:
        transform += [transforms.RandomCrop(28, padding=4),
                      transforms.RandomHorizontalFlip()]
    transform += [transforms.ToTensor()]
    transform += [transforms.Normalize(mean=BLOODMNIST_MEAN,
                                       std=BLOODMNIST_STD)]
    return transforms.Compose(transform)


def bloodmnist_identity_transform() -> transforms:
    """
    Define a basic transformation for CIFAR-100 dataset with only normalization.

    Returns:
    - transforms: A composed transform with only normalization.
    """
    # Define a list of transformations
    transform = [
        transforms.ToTensor(),                 # Convert image to tensor
        # Normalize with mean and std values
        transforms.Normalize(mean=BLOODMNIST_MEAN, std=BLOODMNIST_STD)
    ]
    # Compose the list of transformations into a single transform
    # transforms.Compose is used to chain a list of transformations
    # into a single transformation. The transformations are applied in
    # the order they are listed in the 'transform' variable.
    return transforms.Compose(transform)

# # Example usage
# train_transform = bloodmnist_transform(train=True)
# test_transform = bloodmnist_transform(train=False)

# train_loader, test_loader, val_loader = get_bloodmnist_dataloaders(
#     root='./data', train_transform=train_transform, test_transform=test_transform)

# # Verify the data loading
# for images, labels in train_loader:
#     print(f'Images batch shape: {images.size()}')
#     print(f'Labels batch shape: {labels.size()}')
#     break
