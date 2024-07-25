"""
2024/07/09
Haoyu Shen
-------
"""
import argparse 
import csv 
from dataset.cifar100 import get_cifar100_dataloaders, cifar100_crop_flip_transform
from dataset.cifar10 import get_cifar10_dataloaders, cifar10_crop_flip_transform

import time 
import torch 
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms 
import numpy as np
from  models import model_dict 
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

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed",
                        type = int,
                        default = 34,
                        help = "The random seed.")
    
    parser.add_argument("--batch_size",
                        type = int,
                        default = 64,
                        help = "The batch size used for training.")
    
    parser.add_argument("--print_freq",
                        type = int,
                        default = 100,
                        help = "The print frequency.")
    # dataset
    parser.add_argument("--dataset",
                        type = str,
                        default = "CIFAR100",
                        help = "CIFAR10/CIFAR100/Imagenet, Specify the dataset to train on.")

    parser.add_argument("--dataset_path",
                        type=  str,
                        default = "../../dataset/pytorch_datasets",
                        help = "The place where dataset is stored.")
    
    # training stuff
    parser.add_argument("--model_name",
                        type = str,
                        default = "vgg13",
                        help = "TODO:CHANGE THIS TO CHOICE!")
    
    parser.add_argument("--epoch",
                        type = int,
                        default = 240,
                        help = "Training epoch."
                        )
    # optimizer 
    parser.add_argument("--optimizer",
                        type = str,
                        default = "SGD",
                        help = "The optimizer type.")
    
    parser.add_argument("--lr",
                        type = float,
                        default = 0.05,
                        help = "The learning rate for optimizer.")
    
    parser.add_argument("--weight_decay",
                        type = float,
                        default =  5e-4,
                        help = "The weight decay(L2 regularization term) for optimizer.")
    
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "The momentum for optimizer.")
    
    parser.add_argument('--lr_decay_epochs', 
                        type=str, 
                        default='150,180,210', 
                        help='Where to decay lr, can be a list.')
    
    parser.add_argument('--lr_decay_rate', 
                        type=float, 
                        default=10, 
                        help='decay rate for learning rate.')
    
    # saving stuff

    parser.add_argument("--model_saving_path",
                        type = str,
                        default = "../../dataset/model_zoo/pretrained_teacher",
                        help = "The place for saving optimized model.")
    
    parser.add_argument("--logging_saving_path",
                        type = str,
                        default = "../../dataset/logging",
                        help = "The place for saving logging.")
    
    parser.add_argument('--save_freq', 
                        type = int, 
                        default = 40, 
                        help = 'Saving frequency')

    parser.add_argument('--aug',
                        type = str,
                        default = None,
                        help = "Augmentation type")
    opt = parser.parse_args()
    return opt 

"""From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964"""
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(opt: argparse.Namespace):
    #======= configure logging =======
    logger = logging.getLogger(name = __name__)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream = sys.stdout)
    formatter = logging.Formatter("%(message)s")
    stdout_handler.setFormatter(formatter)
    os.makedirs(opt.logging_saving_path, exist_ok = True)
    file_handler = logging.FileHandler(os.path.join(opt.logging_saving_path,f"log_{opt.model_name}_{opt.dataset}.txt"),"w+")
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.info(opt)
    

    seed = opt.seed 
    seed_everything(seed)
    timer = Timer(opt.epoch)
    best_acc = 0
    best_acc_epoch = 0
    
    config = {
       "task": "pretraining",
       "seed": opt.seed,
       "dataset":opt.dataset,
       "model":opt.model_name,
       "epoch": opt.epoch,
       "lr" : opt.lr,
       "weight_decay": opt.weight_decay,
       "momentum" : opt.momentum,
       "batch_size" : opt.batch_size,
       "time" : datetime.datetime.now()
    }
    wandb.init(project = f"Pretraining Teacher Model {opt.model_name} {opt.dataset}",
               config = config)

    if opt.dataset == "CIFAR100":
        n_cls = 100 
        img_size = 32
        simple_transform = cifar100_crop_flip_transform()
        train_loader, test_loader = get_cifar100_dataloaders(opt.dataset_path,
                                                             train_transform = simple_transform,
                                                             test_transform = simple_transform,
                                                             batch_size = opt.batch_size,
                                                             num_workers=8)
    elif opt.dataset == "CIFAR10":
        n_cls = 10 
        img_size = 32
        simple_transform = cifar10_crop_flip_transform()
        train_loader, test_loader = get_cifar10_dataloaders(opt.dataset_path,
                                                             train_transform = simple_transform,
                                                             test_transform = simple_transform,
                                                             batch_size = opt.batch_size,
                                                             num_workers=8)
    else:
        raise NotImplementedError(opt.dataset)

    model = model_dict[opt.model_name](num_classes = n_cls, img_size = img_size)

    #======= Set up the optimizer =======   
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params = model.parameters(),    
            lr = opt.lr,
            weight_decay = opt.weight_decay,
            momentum = opt.momentum
        )
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = opt.lr,
            weight_decay = opt.weight_decay,
        )
    else:
        raise NotImplementedError(opt.optimizer)
    
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to("cuda")
        criterion = criterion.cuda()
        cudnn.benchmark = True
    os.makedirs(opt.model_saving_path,exist_ok = True)
    #======= training! =======
    for epoch in range(1, opt.epoch + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        
        logger.info("==> Set lr %s @ Epoch %d " % (lr, epoch))

        time1 = time.time()

        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(test_loader, model, criterion, opt)
        
    
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
            save_model_dir  = os.path.join(opt.model_saving_path, opt.model_name)
            os.makedirs(save_model_dir, exist_ok = True)

            save_model = os.path.join(save_model_dir, '{}_{}_seed_{}best.pth'.format(opt.model_name,
                                                                                    opt.dataset,
                                                                                    opt.seed))
            print('saving the best model!')
            torch.save(state, save_model)
        wandb.log({
            "Acc1":test_acc,
            "Acc5":test_acc_top5,
            "Test loss": test_loss,
            "Best_Acc1":best_acc 
        })
        logger.info("Acc1 %.4f Acc5 %.4f TestLoss %.4f Epoch %d (after update) lr %s (Best_Acc1 %.4f @Epoch %d)" % 
            (test_acc, test_acc_top5, test_loss, epoch, lr, best_acc, best_acc_epoch))
        print('Predicted finish time: %s' % timer())

        # Regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict()
            }

            save_ckpt = os.path.join(save_model_dir, '{}_{}_seed_{}_ckpt_epoch_{}.pth'.format(opt.model_name,
                                                                                             opt.dataset,
                                                                                             opt.seed,
                                                                                             epoch))
            torch.save(state, save_ckpt)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    
    csv_path = os.path.join(save_model_dir, 'result.csv')
    try:
        pd.read_csv(csv_path)
    except:
       with open(csv_path,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["time", "seed", "epoch", "lr", "weight_decay", "dataset", "model_name", "best_acc"])
    with open(csv_path, 'a+', newline='') as csvfile:
      # Your CSV writing operations here
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([datetime.datetime.now(), opt.seed, opt.epoch, opt.lr, opt.weight_decay, opt.dataset, opt.model_name,best_acc.cpu().detach().item()]) 
    wandb.finish()

if  __name__ == "__main__":
    opt = get_args()
    main(opt)


# TODO 2024/07/11
# ADD csv to store accuracy_score
# ADD wandb to track experiment 
# TRAIN wrn model 