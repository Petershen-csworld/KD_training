import argparse
import csv
from dataset.cifar100 import get_cifar100_dataloaders, cifar100_crop_flip_transform
from dataset.cifar10 import get_cifar10_dataloaders, cifar10_crop_flip_transform
import datetime
from kd_losses import *
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from models import model_dict
import pandas as pd
from helper.util import adjust_learning_rate
from helper.loops import train_distill, validate_distill
from utils import Timer, seed_everything, load_pretrained_model
import random
import logging
import os
import sys
import wandb
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument("--seed",
                        type=int,
                        default=2023,
                        help="The random seed.")

    # Batch size for training
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="The batch size used for training.")

    # Print frequency during training
    parser.add_argument("--print_freq",
                        type=int,
                        default=100,
                        help="The print frequency.")

    # Dataset selection
    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR10",
                        help="CIFAR10/CIFAR100/Imagenet, Specify the dataset to train on.")

    # Dataset path
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/pytorch_datasets",
                        help="The place where dataset is stored.")

    # Teacher model selection
    parser.add_argument("--t_model",
                        type=str,
                        default="vgg13",
                        help="TODO:CHANGE THIS TO CHOICE!")

    # Student model selection
    parser.add_argument("--s_model",
                        type=str,
                        default="vgg8",
                        help="TODO:CHANGE THIS TO CHOICE!")

    # Path to pretrained teacher model
    parser.add_argument("--t_init",
                        type=str,
                        default="/home/shenhaoyu/dataset/model_zoo/pretrained_teacher/vgg13/vgg13_CIFAR10_seed_3407best.pth",
                        help="Path of teacher model.")

    # Number of training epochs
    parser.add_argument("--epoch",
                        type=int,
                        default=240,
                        help="Training epoch."
                        )

    # Optimizer type
    parser.add_argument("--optimizer",
                        type=str,
                        default="SGD",
                        help="The optimizer type.")

    # Learning rate
    parser.add_argument("--lr",
                        type=float,
                        default=0.05,
                        help="The learning rate for optimizer.")

    # Weight decay for optimizer
    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="The weight decay(L2 regularization term) for optimizer.")

    # Momentum for optimizer
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="The momentum for optimizer.")

    # Learning rate decay epochs
    parser.add_argument('--lr_decay_epochs',
                        type=str,
                        default='150,180,210',
                        help='Where to decay lr, can be a list.')

    # Learning rate decay rate
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=10,
                        help='decay rate for learning rate.')

    # Model saving path
    parser.add_argument("--model_saving_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/model_zoo/distilled_student_orig",
                        help="The place for saving optimized model.")

    # Logging saving path
    parser.add_argument("--logging_saving_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/logging",
                        help="The place for saving logging.")

    # Model saving frequency
    parser.add_argument('--save_freq',
                        type=int,
                        default=40,
                        help='Saving frequency')

    # Knowledge Distillation mode
    parser.add_argument('--kd_mode', type=str, default="st", help='mode of kd, which can be:'
                        'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                        'sp/sobolev/cc/lwm/irg/vid/ofd/afd')

    # Trade-off parameter for KD loss
    parser.add_argument('--lambda_kd', type=float, default=0.9,
                        help='trade-off parameter for kd loss')

    # Temperature for Soft Target
    parser.add_argument('--T', type=float, default=4.0,
                        help='temperature for ST')

    # Augmentation for input
    parser.add_argument('--aug',
                        type=str,
                        default=None,
                        help="The augmentation for input.")

    opt = parser.parse_args()
    return opt


"""From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964"""

# Function to set all seeds for reproducibility


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(opt: argparse.Namespace):
    # Initialize empty lists for logging purposes
    opt.prob = []
    opt.entropy = []
    opt.all_avg_prob = []
    torch.backends.cudnn.benchmark = True

    # ======= Configure logging =======
    logger = logging.getLogger(name=__name__)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(message)s")
    stdout_handler.setFormatter(formatter)
    os.makedirs(opt.logging_saving_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        opt.logging_saving_path, f"log_{opt.t_model}_{opt.s_model}_{opt.dataset}.txt"), "w+")
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.info(opt)

    # Configure WandB for experiment tracking
    project_name = f"{opt.dataset}_Pure_softtarget_T_{opt.t_model}_S_{opt.s_model}"
    config = {
        "task": "KD",
        "seed": opt.seed,
        "dataset": opt.dataset,
        "t_model": opt.t_model,
        "s_model": opt.s_model,
        "epoch": opt.epoch,
        "lr": opt.lr,
        "weight_decay": opt.weight_decay,
        "momentum": opt.momentum,
        "batch_size": opt.batch_size,
        "time": datetime.datetime.now()
    }
    wandb.init(project=project_name, config=config)

    # Set the random seed
    seed = opt.seed
    seed_everything(seed)

    # Initialize the timer
    timer = Timer(opt.epoch)

    best_acc = 0
    best_acc_epoch = 0

    # Dataset preparation
    if opt.dataset == "CIFAR100":
        n_cls = 100
        img_size = 32
        simple_transform = cifar100_crop_flip_transform()
        train_loader, test_loader = get_cifar100_dataloaders(opt.dataset_path,
                                                             train_transform=simple_transform,
                                                             test_transform=simple_transform,
                                                             batch_size=opt.batch_size,
                                                             num_workers=8)
    elif opt.dataset == "CIFAR10":
        n_cls = 10
        img_size = 32
        simple_transform = cifar10_crop_flip_transform()
        train_loader, test_loader = get_cifar10_dataloaders(opt.dataset_path,
                                                            train_transform=simple_transform,
                                                            test_transform=simple_transform,
                                                            batch_size=opt.batch_size,
                                                            num_workers=8)
    else:
        raise NotImplementedError(opt.dataset)

    # ======= Load the pretrained model =======
    t_model = model_dict[opt.t_model](num_classes=n_cls, img_size=img_size)
    t_checkpoint = torch.load(opt.t_init)
    load_pretrained_model(t_model, t_checkpoint['model'])

    s_model = model_dict[opt.s_model](num_classes=n_cls, img_size=img_size)

    # ======= Disable the parameters in teacher model =======
    for param in t_model.parameters():
        param.requires_grad = False

    # ======= Set up the optimizer =======
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params=s_model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            momentum=opt.momentum,
            nesterov=True
        )
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=s_model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
    else:
        raise NotImplementedError(opt.optimizer)

    # Define loss functions
    if opt.kd_mode == "logits":
        criterionKD = Logits()
    elif opt.kd_mode == "st":
        criterionKD = SoftTarget(T=opt.T)
    else:
        raise Exception('Invalid kd mode...')

    criterionCls = torch.nn.CrossEntropyLoss().cuda()
    if torch.cuda.is_available():
        t_model = t_model.cuda()
        s_model = s_model.cuda()
        criterionCls = criterionCls.cuda()
        criterionKD = criterionKD.cuda()
        cudnn.benchmark = True

    # Create directories for saving models
    os.makedirs(opt.model_saving_path, exist_ok=True)

    # Dictionary of models and criteria
    nets = {"snet": s_model, "tnet": t_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    # ======= Training =======
    for epoch in range(1, opt.epoch + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        # Log the current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        logger.info("==> Set lr %s @ Epoch %d " % (lr, epoch))

        # Training step
        time1 = time.time()
        train_acc1, train_acc5, train_loss_cls, train_loss_kd, std, entropy = train_distill(
            epoch, train_loader, nets, criterions, optimizer, opt)
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Validation step
        test_acc, test_acc_top5, cls_loss, kd_loss = validate_distill(
            test_loader, nets, criterions, opt)

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch
            state = {
                'epoch': epoch,
                'model': s_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_model_dir = os.path.join(opt.model_saving_path, opt.s_model)
            os.makedirs(save_model_dir, exist_ok=True)
            save_file = os.path.join(save_model_dir, 'T_{}_S_{}_{}_seed_{}best.pth'.format(opt.t_model,
                                                                                           opt.s_model,
                                                                                           opt.dataset,
                                                                                           opt.seed))
            print('saving the best model!')
            torch.save(state, save_file)

        # Log metrics to WandB
        wandb.log({
            "test_acc": test_acc,
            "Acc1": test_acc,
            "Acc5": test_acc_top5,
            "Best_Acc1": best_acc,
            'T.stddev': std,
            'Entropy': entropy
        })
        logger.info("Acc1 %.4f Acc5 %.4f TestClsLoss %.4f TestKDLoss %.4f Epoch %d (after update) lr %s (Best_Acc1 %.4f @Epoch %d)" %
                    (test_acc, test_acc_top5, cls_loss, kd_loss, epoch, lr, best_acc, best_acc_epoch))
        print('Predicted finish time: %s' % timer())

        # Regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': s_model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict()
            }
            save_model_dir = os.path.join(opt.model_saving_path, opt.s_model)
            os.makedirs(save_model_dir, exist_ok=True)
            save_ckpt = os.path.join(save_model_dir, 'T_{}_S_{}_{}_seed_{}_ckpt_epoch_{}.pth'.format(opt.t_model,
                                                                                                     opt.s_model,
                                                                                                     opt.dataset,
                                                                                                     opt.seed,
                                                                                                     epoch))
            # Save the current state of the model, optimizer, and other training parameters to a file.
            # This allows for saving the training progress and resuming it later from this point.

            # `torch.save` uses Python's `pickle` module to serialize the given object (`state`)
            # and save it to the specified file path (`save_ckpt`). The `state` object typically includes:
            # - model's state dictionary (model's parameters)
            # - optimizer's state dictionary (optimizer's parameters and buffers)
            # - other relevant training information (e.g., current epoch, loss)

            torch.save(state, save_ckpt)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # Save results to a CSV file
    # Define the path to the CSV file
    csv_path = os.path.join(save_model_dir, 'result.csv')

    try:
        # Try to read the CSV file to check if it already exists
        pd.read_csv(csv_path)
    except:
        # If the CSV file does not exist, create it and write the header row
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row with the names of the columns
            writer.writerow(["time", "seed", "epoch", "lr", "weight_decay",
                            "dataset", "temperature", "lambdaKD", "T_model_name", "best_acc"])

    # Open the CSV file in append mode
    with open(csv_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write a new row with the current results
        csvwriter.writerow([datetime.datetime.now(),  # Current time
                            opt.seed,                # Seed value used in the experiment
                            opt.epoch,               # Current epoch number
                            opt.lr,                  # Learning rate
                            opt.weight_decay,        # Weight decay value
                            opt.dataset,             # Dataset name
                            opt.T,                   # Temperature value for softmax
                            opt.lambda_kd,           # Lambda value for knowledge distillation
                            opt.t_model,             # Teacher model name
                            # Best accuracy, detached from GPU and converted to a Python number
                            best_acc.cpu().detach().item()
                            ])


if __name__ == "__main__":
    opt = get_args()
    main(opt)
