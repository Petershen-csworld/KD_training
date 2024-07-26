import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision 
from typing import Tuple 
import torchvision.transforms as transforms 
from torchvision.datasets import CIFAR10
import os 


""" Taken from: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data"""
MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.24703233, 0.24348505, 0.26158768]

def get_cifar10_dataloaders(root:str = "./data",
                             train_transform:transforms.transforms = None,
                             test_transform:transforms.transforms = None,
                             batch_size:int = 128,
                             num_workers:int = 8
                             ) -> Tuple[DataLoader, DataLoader]:
    if not os.path.exists(root):
       os.makedirs(root, exist_ok = True)
    train_ds = CIFAR10(root = root, transform = train_transform, download = True, train = True)
    test_ds = CIFAR10(root = root, transform = test_transform, download = True, train = False)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, 
                              pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = True, num_workers = 1 if num_workers == 1 else num_workers//2, 
                             pin_memory = True)
    return train_loader, test_loader 


def cifar10_crop_flip_transform() -> transforms:
    transform = [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]
    transform += [transforms.ToTensor()]
    transform += [transforms.Normalize(mean = MEAN, std = STD)]
    return transforms.Compose(transform)
    
def cifar10_simple_transform() -> transforms:
    transform = [transforms.ToTensor()]
    transform += [transforms.Normalize(mean = MEAN, std = STD)]
    return transforms.Compose(transform)

