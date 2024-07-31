import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from typing import Tuple
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import os


""" Taken from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151"""
MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]


def get_cifar100_dataloaders(
    root: str = "./data",
    train_transform: transforms.transforms = None,
    test_transform: transforms.transforms = None,
    batch_size: int = 128,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders for CIFAR-100 dataset.

    Parameters:
    - root (str): Path to the directory where dataset will be stored.
    - train_transform (transforms.transforms): Transformations to apply to the training data.
    - test_transform (transforms.transforms): Transformations to apply to the testing data.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - Tuple[DataLoader, DataLoader]: Training and test data loaders.
    """
    # Create the directory if it does not exist
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    # Load CIFAR-100 dataset with specified transformations
    train_ds = CIFAR100(root=root, transform=train_transform,
                        download=True, train=True)
    test_ds = CIFAR100(root=root, transform=test_transform,
                       download=True, train=False)

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data
        num_workers=num_workers,  # Number of subprocesses for data loading
        pin_memory=True        # Pin memory for faster data transfer to GPU
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,         # Do not shuffle test data
        # Use fewer workers for test data
        num_workers=1 if num_workers == 1 else num_workers // 2,
        pin_memory=True        # Pin memory for faster data transfer to GPU
    )

    return train_loader, test_loader


def cifar100_crop_flip_transform() -> transforms:
    """
    Define transformations for CIFAR-100 dataset that include random cropping and horizontal flipping.

    Returns:
    - transforms: A composed transform with random cropping, horizontal flipping, and normalization.
    """
    # Define a list of transformations
    transform = [
        # Randomly crop the image with padding
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()      # Randomly flip the image horizontally
    ]
    # Convert image to tensor and normalize
    transform += [
        transforms.ToTensor(),                 # Convert image to tensor
        # Normalize with mean and std values
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    # Compose the list of transformations into a single transform
    return transforms.Compose(transform)


def cifar100_identiy_transform() -> transforms:
    """
    Define a basic transformation for CIFAR-100 dataset with only normalization.

    Returns:
    - transforms: A composed transform with only normalization.
    """
    # Define a list of transformations
    transform = [
        transforms.ToTensor(),                 # Convert image to tensor
        # Normalize with mean and std values
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    # Compose the list of transformations into a single transform
    # transforms.Compose is used to chain a list of transformations
    # into a single transformation. The transformations are applied in
    # the order they are listed in the 'transform' variable.
    return transforms.Compose(transform)



# import medmnist
# from medmnist import INFO
# from torchvision import transforms
# from torch.utils.data import DataLoader

# # 选择BloodMNIST数据集
# data_flag = 'bloodmnist'
# download = True

# # 指定下载路径
# custom_path = '/path/to/custom/directory'  # 替换为你想要的下载路径

# # 获取数据集信息
# info = INFO[data_flag]
# DataClass = getattr(medmnist, info['python_class'])

# # 定义数据转换
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# # 加载训练集
# train_dataset = DataClass(split='train', transform=data_transform, download=download, root=custom_path)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# # 加载测试集
# test_dataset = DataClass(split='test', transform=data_transform, download=download, root=custom_path)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# # 加载验证集
# val_dataset = DataClass(split='val', transform=data_transform, download=download, root=custom_path)
# val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# # 验证数据加载
# for images, labels in train_loader:
#     print(f'Images batch shape: {images.size()}')
#     print(f'Labels batch shape: {labels.size()}')
#     break
