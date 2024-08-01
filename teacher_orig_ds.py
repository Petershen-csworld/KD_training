import argparse
import csv
from dataloader.cifar100 import get_cifar100_dataloaders, cifar100_crop_flip_transform
from dataloader.cifar10 import get_cifar10_dataloaders, cifar10_crop_flip_transform
from dataloader.bloodMNIST import get_bloodmnist_dataloaders, bloodmnist_crop_flip_transform

import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from models import model_dict
from helper.util import adjust_learning_rate
from helper.loops import train_vanilla as train, validate
from utils import Timer, seed_everything
import random
import logging
import os
import sys
import wandb
import pandas as pd
import datetime
from args import get_args
from tqdm import tqdm


def main(args: argparse.Namespace):
    # ======= configure logging =======
    logger = logging.getLogger(name=__name__)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(message)s")
    stdout_handler.setFormatter(formatter)
    os.makedirs(args.logging_saving_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        args.logging_saving_path, f"log_{args.model_name}_{args.dataset}.txt"), "w+")
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.info(args)

    # Set random seed
    seed = args.seed
    seed_everything(seed)

    # Timer for estimating training time
    timer = Timer(args.epoch)
    best_acc = 0
    best_acc_epoch = 0

    # Configuring wandb for tracking experiments
    config = {
        "task": "pretraining",
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model_name,
        "epoch": args.epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "time": datetime.datetime.now()
    }
    wandb.init(project=f"Pretraining Teacher Model {args.model_name} {args.dataset}",
               entity="windyyxz-Xihu University",
               config=config)
    args.device = torch.device(f"cuda:{args.gpu}")  # 假设你想要使用索引为1的GPU

    # Load dataset and transformations
    if args.dataset == "CIFAR100":
        n_cls = 100
        img_size = 32
        simple_transform = cifar100_crop_flip_transform()
        train_loader, test_loader = get_cifar100_dataloaders(args.dataset_path,
                                                             train_transform=simple_transform,
                                                             test_transform=simple_transform,
                                                             batch_size=args.batch_size,
                                                             num_workers=8)
    elif args.dataset == "CIFAR10":
        n_cls = 10
        img_size = 32
        simple_transform = cifar10_crop_flip_transform()
        train_loader, test_loader = get_cifar10_dataloaders(args.dataset_path,
                                                            train_transform=simple_transform,
                                                            test_transform=simple_transform,
                                                            batch_size=args.batch_size,
                                                            num_workers=8)
    elif args.dataset == "BloodMNIST":
        n_cls = 8
        img_size = 28
        simple_transform = bloodmnist_crop_flip_transform()
        train_loader, test_loader = get_bloodmnist_dataloaders(args.dataset_path,
                                                               train_transform=simple_transform,
                                                               test_transform=simple_transform,
                                                               batch_size=args.batch_size,
                                                               num_workers=8)

    else:
        raise NotImplementedError(args.dataset)

    # Initialize model from the model dictionary
    model = model_dict[args.model_name](num_classes=n_cls, img_size=img_size)

    # ======= Set up the optimizer =======
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(args.optimizer)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Move model and criterion to GPU if available
    if torch.cuda.is_available():
        model = model.to(args.device)
        criterion = criterion.to(args.device)
        cudnn.benchmark = True

    # Create directory for saving models if it does not exist
    os.makedirs(args.model_saving_path, exist_ok=True)

    # ======= training! =======
    for epoch in tqdm(range(1, args.epoch + 1), desc='training'):
        # Adjust learning rate according to schedule
        adjust_learning_rate(epoch, args, optimizer)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        logger.info("==> Set lr %s @ Epoch %d " % (lr, epoch))

        time1 = time.time()

        # Train for one epoch
        train_acc, train_loss = train(
            epoch, train_loader, model, criterion, optimizer, args)
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Validate the model
        test_acc, test_acc_top5, test_loss = validate(
            test_loader, model, criterion, args)

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            save_model_dir = os.path.join(
                args.model_saving_path, args.model_name)
            os.makedirs(save_model_dir, exist_ok=True)

            save_model = os.path.join(save_model_dir, '{}_{}_seed_{}best.pth'.format(args.model_name,
                                                                                     args.dataset,
                                                                                     args.seed))
            print('saving the best model!')
            torch.save(state, save_model)

        # Log metrics to wandb
        wandb.log({
            "Acc1": test_acc,
            "Acc5": test_acc_top5,
            "Test loss": test_loss,
            "Best_Acc1": best_acc
        })
        logger.info("Acc1 %.4f Acc5 %.4f TestLoss %.4f Epoch %d (after update) lr %s (Best_Acc1 %.4f @Epoch %d)" %
                    (test_acc, test_acc_top5, test_loss, epoch, lr, best_acc, best_acc_epoch))
        print('Predicted finish time: %s' % timer())

        # Regular saving
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict()
            }

            save_ckpt = os.path.join(save_model_dir, '{}_{}_seed_{}_ckpt_epoch_{}.pth'.format(args.model_name,
                                                                                              args.dataset,
                                                                                              args.seed,
                                                                                              epoch))
            torch.save(state, save_ckpt)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # Save results to a CSV file
    csv_path = os.path.join(save_model_dir, 'result.csv')
    try:
        pd.read_csv(csv_path)
    except:
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time", "seed", "epoch", "lr",
                            "weight_decay", "dataset", "model_name", "best_acc"])
    with open(csv_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([datetime.datetime.now(), args.seed, args.epoch, args.lr,
                           args.weight_decay, args.dataset, args.model_name, best_acc.cpu().detach().item()])

    # Finish wandb logging
    wandb.finish()


# Entry point of the script
if __name__ == "__main__":
    args = get_args()
    main(args)
