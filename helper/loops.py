from __future__ import division, print_function

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import AverageMeter, accuracy


def mixup_data(x, y, args, alpha=0.4):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(args.gpu)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix(data, targets, alpha=0.25):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    # shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)

    return data, None


def train_vanilla(epoch, train_loader, model, criterion, optimizer, args):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.to(args.device)
            target = target.to(args.device)

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # Print
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

            sys.stdout.flush()
    args.prob = []
    print(args.prob)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(args.device)
                target = target.to(args.device)

            # compute output
            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg, losses.avg


# Function to train the model using knowledge distillation
def train_distill(epoch, train_loader, nets, criterions, optimizer, args, map_dict=None):
    batch_time = AverageMeter()  # Initialize meter to track batch processing time
    data_time = AverageMeter()   # Initialize meter to track data loading time
    cls_losses = AverageMeter()  # Initialize meter to track classification losses
    kd_losses = AverageMeter()   # Initialize meter to track knowledge distillation losses
    top1 = AverageMeter()        # Initialize meter to track top-1 accuracy
    top5 = AverageMeter()        # Initialize meter to track top-5 accuracy

    snet = nets['snet']  # Student network
    tnet = nets['tnet']  # Teacher network

    # Classification loss function
    criterionCls = criterions['criterionCls']

    # Knowledge distillation loss function
    criterionKD = criterions['criterionKD']

    snet.train()  # Set the student network to training mode
    end = time.time()  # Record the current time

    # Iterate over the training data
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)  # Update data loading time
        img = img.cuda(non_blocking=True)  # Move images to GPU
        target = target.cuda(non_blocking=True)  # Move targets to GPU

        out_s = snet(img)  # Forward pass through student network
        out_t = tnet(img)  # Forward pass through teacher network

        if args.aug is None:  # If no augmentation is specified
            # Compute the classification loss between the student network's output and the target labels
            cls_loss = criterionCls(out_s, target)

            # Check the knowledge distillation mode and compute the corresponding KD loss
            if args.kd_mode in ['logits', 'st']:
                # Compute KD loss with detached teacher output

                # Create a new tensor that shares the same data as the original tensor,
                # but does not compute gradients during backpropagation
                kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
            else:
                # Raise an exception if an invalid KD mode is specified
                raise Exception('Invalid kd mode...')

            # Combine the classification loss and KD loss using the lambda_kd weight
            loss = (1 - args.lambda_kd) * cls_loss + kd_loss
        else:  # If augmentation is specified
            if args.aug == "diffmix":  # Check if the augmentation method is "diffmix"
                # Convert target labels to numpy array
                label_arr = target.cpu().numpy()

                # Map the target labels to two different sets of labels for "diffmix"
                label_1, label_2 = torch.LongTensor([map_dict[i][0] for i in label_arr]).cuda(
                ), torch.LongTensor([map_dict[i][1] for i in label_arr]).cuda()

                lam = 0.5  # Set lambda value for mixing

                # Compute the classification loss using the first set of mapped labels
                cls_loss = criterionCls(out_s, label_1)

                # Check the knowledge distillation mode and compute the corresponding KD loss
                if args.kd_mode in ['logits', 'st']:
                    # Compute KD loss with detached teacher output
                    kd_loss = criterionKD(
                        out_s, out_t.detach()) * args.lambda_kd

                    # Combine the classification loss and KD loss using the lambda_kd weight
                    loss = (1 - args.lambda_kd) * cls_loss + kd_loss
                else:
                    # Raise an exception if an invalid KD mode is specified
                    raise Exception('Invalid kd mode...')

        prec1, prec5 = accuracy(
            out_s, target, topk=(1, 5))  # Compute accuracies

        # Update classification loss
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))  # Update KD loss
        top1.update(prec1.item(), img.size(0))  # Update top-1 accuracy
        top5.update(prec5.item(), img.size(0))  # Update top-5 accuracy

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

        batch_time.update(time.time() - end)  # Update batch processing time
        end = time.time()  # Record the current time

        # Print log information at specified intervals
        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                       'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                           epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                           cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
            print(log_str)

        # Calculate and log the probabilities and entropy for the teacher model

        # Logits are the unnormalized output values of the last layer (usually a fully connected layer) of
        # a neural network model. They are the raw scores before applying activation functions (such as softmax).
        # These logits can be used to calculate the probability output and loss value of the model.
        logit_t = out_t  # Get the teacher model logits
        # Compute the softmax probabilities from the logits
        p = F.softmax(logit_t, dim=-1)
        # Store the computed probabilities in the args.prob list
        args.prob += [p]
        # Compute and store the entropy values for each probability distribution
        args.entropy += [(-p * torch.log(p)).sum(dim=-1)]

        num = 10  # Number of samples to use for calculating statistics
        if len(args.prob) >= num:
            # Concatenate all stored probabilities along the batch dimension
            prob = torch.cat(args.prob, dim=0)
            # Concatenate all stored entropy values along the batch dimension
            entropy = torch.cat(args.entropy, dim=0)
            # Calculate the average probability distribution
            avg_prob = prob.mean(dim=0)
            # Store the average probability distribution in args.all_avg_prob
            args.all_avg_prob += [avg_prob]
            args.prob = []  # Reset the probability list for the next batch of samples

            # Stack all average probabilities into a tensor
            all_avg_prob = torch.stack(args.all_avg_prob, dim=0)
            # Compute the standard deviation of the average probabilities
            avg_prob_std = all_avg_prob.std(dim=0)
            # Format the mean standard deviation as a string
            std_str = '%.6f' % avg_prob_std.mean().item()
            if i % args.print_freq == 0:  # Check if the current iteration is at the specified print frequency
                # Print the statistics including the number of sampled standard deviations, epoch, mean standard deviation, and mean entropy
                print(
                    f'Check T prob: NumOfSampledStd {len(args.all_avg_prob)} Epoch {epoch}  MeanStd {std_str} MeanEntropy {entropy.mean().item():.6f}')

    # Return the average metrics
    return top1.avg, top5.avg, cls_losses.avg, kd_losses.avg, avg_prob_std.mean().item(), entropy.mean().item()


def validate_distill(test_loader, nets, criterions, args):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD = criterions['criterionKD']

    snet.eval()
    end = time.time()
    for _, (img, target) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()

        out_s = snet(img)
        out_t = tnet(img)
        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        else:
            raise Exception('Invalid kd mode...')

        prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
    print('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg, cls_losses.avg, kd_losses.avg
