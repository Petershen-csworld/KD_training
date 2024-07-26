import numpy as np
import torch
from torch.optim import Optimizer
from utils import get_lr


def adjust_learning_rate(epoch, opt, optimizer):
    """
    Adjusts the learning rate of the optimizer based on the current epoch.

    This function checks if the current epoch is one of the epochs where the learning rate should be adjusted.
    If it is, then the learning rate is decayed by the specified decay rate.

    Parameters:
    - epoch (int): The current training epoch.
    - opt (argparse.Namespace): An object containing the command-line arguments or configuration options.
      It must have an attribute `lr_decay_epochs`, which is a comma-separated string of epochs at which to decay the learning rate,
      and `lr_decay_rate`, which is the factor by which to decay the learning rate.
    - optimizer (Optimizer): The optimizer whose learning rate will be adjusted.
    """
    # Convert the comma-separated string of decay epochs into a list of integers
    epoch_array = [int(e) for e in opt.lr_decay_epochs.split(",")]

    # Check if the current epoch is in the list of decay epochs
    if epoch in epoch_array:
        # Get the current learning rate
        current_lr = get_lr(optimizer)

        # Calculate the new learning rate by dividing the current learning rate by the decay rate
        new_lr = current_lr / opt.lr_decay_rate

        # Set the new learning rate for each parameter group in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def get_lr(optimizer):
    """
    Returns the learning rate of the first parameter group in the optimizer.

    Parameters:
    - optimizer (Optimizer): The optimizer from which to retrieve the learning rate.

    Returns:
    - float: The learning rate of the first parameter group.
    """
    # Retrieve the learning rate from the first parameter group
    for group in optimizer.param_groups:
        lr = group['lr']
        return lr
