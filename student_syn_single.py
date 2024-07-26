import argparse
import csv
from dataset.cifar100 import get_cifar100_dataloaders, cifar100_crop_flip_transform, cifar100_identiy_transform
from dataset.cifar10 import get_cifar10_dataloaders, cifar10_simple_transform
from dataset.synthetic import get_synset_dataloader_fulltrain, syn_to_cifar_transform
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

    parser.add_argument("--seed",
                        type=int,
                        default=2023,
                        help="The random seed.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="The batch size used for training.")

    parser.add_argument("--print_freq",
                        type=int,
                        default=100,
                        help="The print frequency.")
    # dataset
    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR10",
                        help="CIFAR10/CIFAR100/Imagenet, Specify the dataset to train on.")

    parser.add_argument("--dataset_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/generated/generate_results_basic_single_objectCIFAR10",
                        help="The place where dataset is stored.")

    # training stuff
    parser.add_argument("--t_model",
                        type=str,
                        default="vgg13",
                        help="TODO:CHANGE THIS TO CHOICE!")

    parser.add_argument("--s_model",
                        type=str,
                        default="vgg8",
                        help="TODO:CHANGE THIS TO CHOICE!")

    parser.add_argument("--t_init",
                        type=str,
                        default="/home/shenhaoyu/dataset/model_zoo/pretrained_teacher/vgg13/vgg13_CIFAR10_seed_3407best.pth",
                        help="Path of teacher model.")

    parser.add_argument("--epoch",
                        type=int,
                        default=240,
                        help="Training epoch."
                        )
    # optimizer
    parser.add_argument("--optimizer",
                        type=str,
                        default="SGD",
                        help="The optimizer type.")

    parser.add_argument("--lr",
                        type=float,
                        default=0.05,
                        help="The learning rate for optimizer.")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="The weight decay(L2 regularization term) for optimizer.")

    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="The momentum for optimizer.")

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
                        type=str,
                        default="/home/shenhaoyu/dataset/model_zoo/distilled_student_syn",
                        help="The place for saving optimized model.")

    parser.add_argument("--logging_saving_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/logging",
                        help="The place for saving logging.")

    parser.add_argument('--save_freq',
                        type=int,
                        default=40,
                        help='Saving frequency')
    # hyperparameter
    parser.add_argument('--kd_mode', type=str, default="st", help='mode of kd, which can be:'
                        'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                        'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
    parser.add_argument('--lambda_kd', type=float, default=0.9,
                        help='trade-off parameter for kd loss')
    parser.add_argument('--T', type=float, default=4.0,
                        help='temperature for ST')
    parser.add_argument('--aug',
                        type=str,
                        default=None,
                        help="The augmentation for input.")

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
    opt.prob = []
    opt.entropy = []
    opt.all_avg_prob = []
    # ======= configure logging =======
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

    project_name = f"Single_object_syn_softtarget_T_{opt.t_model}_S_{opt.s_model}"
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
    seed = opt.seed
    seed_everything(seed)
    timer = Timer(opt.epoch)
    best_acc = 0
    best_acc_epoch = 0
    if opt.dataset == "CIFAR100":
        n_cls = 100
        img_size = 32
        simple_transform = cifar100_identiy_transform()
        syn_train_transform = syn_to_cifar_transform()
        _, test_loader = get_cifar100_dataloaders("./data",
                                                  train_transform=simple_transform,
                                                  test_transform=simple_transform,
                                                  batch_size=opt.batch_size,
                                                  num_workers=8
                                                  )
        train_loader, class_to_idx = get_synset_dataloader_fulltrain(root=opt.dataset_path,
                                                                     transform=syn_train_transform,
                                                                     batch_size=opt.batch_size,
                                                                     num_workers=8)
    elif opt.dataset == "CIFAR10":
        n_cls = 10
        img_size = 32
        simple_transform = cifar10_simple_transform()
        syn_train_transform = syn_to_cifar_transform()
        _, test_loader = get_cifar10_dataloaders("./data",
                                                 train_transform=simple_transform,
                                                 test_transform=simple_transform,
                                                 batch_size=opt.batch_size,
                                                 num_workers=8
                                                 )
        train_loader, class_to_idx = get_synset_dataloader_fulltrain(root=opt.dataset_path,
                                                                     transform=syn_train_transform,
                                                                     batch_size=opt.batch_size,
                                                                     num_workers=8)
    else:
        raise NotImplementedError(opt.dataset)
    # ======= Load the pretrained model =======
    t_model = model_dict[opt.t_model](num_classes=n_cls, img_size=img_size)
    t_checkpoint = torch.load(opt.t_init)
    load_pretrained_model(t_model, t_checkpoint['model'])

    s_model = model_dict[opt.s_model](num_classes=n_cls, img_size=img_size)
    # =======Disable the parameters in teacher model=======
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
    # define loss functions
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
    os.makedirs(opt.model_saving_path, exist_ok=True)
    nets = {"snet": s_model, "tnet": t_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    # ======= training! =======
    for epoch in range(1, opt.epoch + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        logger.info("==> Set lr %s @ Epoch %d " % (lr, epoch))

        time1 = time.time()
        train_acc1, train_acc5, train_loss_cls, train_loss_kd, std, entropy = train_distill(
            epoch, train_loader, nets, criterions, optimizer, opt)
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
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
                'optimizer': optimizer.state_dict()
            }
            save_model_dir = os.path.join(opt.model_saving_path, opt.s_model)
            os.makedirs(save_model_dir, exist_ok=True)
            save_file = os.path.join(save_model_dir, 'T_{}_S_{}_{}_seed_{}best.pth'.format(opt.t_model,
                                                                                           opt.s_model,
                                                                                           opt.dataset,
                                                                                           opt.seed))
            print('saving the best model!')
            torch.save(state, save_file)
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
            torch.save(state, save_ckpt)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    csv_path = os.path.join(save_model_dir, 'result.csv')
    try:
        pd.read_csv(csv_path)
    except:
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time", "seed", "epoch", "lr", "weight_decay",
                            "dataset", "temperature", "lambdaKD", "T_model_name", "best_acc"])
    with open(csv_path, 'a+', newline='') as csvfile:
      # Your CSV writing operations here
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([datetime.datetime.now(), opt.seed, opt.epoch, opt.lr, opt.weight_decay,
                           opt.dataset, opt.T, opt.lambda_kd, opt.t_model, best_acc.cpu().detach().item()])


if __name__ == "__main__":
    opt = get_args()
    main(opt)
