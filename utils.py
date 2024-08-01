"""
2024/07/09
Haoyu Shen
-------
"""
import time
import os
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import random

class AverageMeter():
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.count += n
        self.val = val
        self.sum += n * val
        self.avg = self.sum / self.count


def get_lr(optimizer: Optimizer) -> torch.dtype:
    for group in optimizer.param_groups:
        lr = group['lr']
        return lr


"""Reference: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
"""


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps *
                            (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - \
                        int(self.first_cycle_steps *
                            (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * \
                        self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # Because of pytorch new versions, this does not work anymore (pt1.3 is okay, pt1.9 not okay).
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Timer():
    '''Log down iteration time and predict the left time for the left iterations
    '''

    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.time_stamp = []

    def predict_finish_time(self, ave_window=3):
        self.time_stamp.append(time.time())  # update time stamp
        if len(self.time_stamp) == 1:
            return 'only one time stamp, not enough to predict'
        interval = []
        for i in range(len(self.time_stamp) - 1):
            t = self.time_stamp[i + 1] - self.time_stamp[i]
            interval.append(t)
        sec_per_epoch = np.mean(interval[-ave_window:])
        left_t = sec_per_epoch * (self.total_epoch - len(interval))
        finish_t = left_t + time.time()
        finish_t = time.strftime('%Y/%m/%d-%H:%M', time.localtime(finish_t))
        total_t = '%.2fh' % ((np.sum(interval) + left_t) / 3600.)
        return finish_t + ' (speed: %.2fs per timing, total_time: %s)' % (sec_per_epoch, total_t)

    def __call__(self):
        return (self.predict_finish_time())


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


class Cutout(object):
    """
    Transform numpy image
    """

    def __init__(self, h, w, length, n_holes=1):
        self.h = h
        self.w = w
        self.n_holes = n_holes
        self.length = length

    def __call__(self):
        """
        Args:
            img (array) : size (H, W, 3)
        """
        h = self.h
        w = self.w

        mask = np.zeros((h, w, 3), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            # clip restrict the image in the range of [a_min, a_max]
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2, :] = 1
        mask = (mask * 255).astype(np.uint8)
        return mask
