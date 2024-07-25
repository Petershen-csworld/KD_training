import numpy as np 
import torch 
from torch.optim import Optimizer 
from utils import get_lr 
def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""

    epoch_array = [int(e) for e in opt.lr_decay_epochs.split(",")]
    if epoch in epoch_array:
        new_lr = get_lr(optimizer) / opt.lr_decay_rate 
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def get_lr(optimizer : Optimizer) -> torch.dtype:
    for group in optimizer.param_groups:
        lr = group['lr']
        return lr 