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

def get_cifar100_dataloaders(root:str = "./data",
                             train_transform:transforms.transforms = None,
                             test_transform:transforms.transforms = None,
                             batch_size:int = 128,
                             num_workers:int = 8
                             ) -> Tuple[DataLoader, DataLoader]:
    if not os.path.exists(root):
       os.makedirs(root, exist_ok = True)
    train_ds = CIFAR100(root = root, transform = train_transform, download = True, train = True)
    test_ds = CIFAR100(root = root, transform = test_transform, download = True, train = False)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, 
                              pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = True, num_workers = 1 if num_workers == 1 else num_workers//2, 
                             pin_memory = True)
    return train_loader, test_loader 


def cifar100_crop_flip_transform() -> transforms:
    transform = [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]
    transform += [transforms.ToTensor()]
    transform += [transforms.Normalize(mean = MEAN, std = STD)]
    return transforms.Compose(transform)
    
def cifar100_identiy_transform() -> transforms:
    transform = [transforms.ToTensor()]
    transform += [transforms.Normalize(mean = MEAN, std = STD)]
    return transforms.Compose(transform)
    