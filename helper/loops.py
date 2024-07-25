from __future__ import division, print_function

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import AverageMeter, accuracy


def mixup_data(x, y, opt, alpha=0.4):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(opt.gpu)

    mixed_x = lam * x + (1 - lam) * x[index,:]
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


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
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
            input = input.cuda()
            target = target.cuda()

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
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            sys.stdout.flush()
    opt.prob = []
    print(opt.prob)
    return top1.avg, losses.avg















def validate(val_loader, model, criterion, opt):
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
                input = input.cuda()
                target = target.cuda()

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




# epoch, train_loader, model, criterion, optimizer, opt
def train_distill(epoch, train_loader, nets,  criterions, optimizer, opt, map_dict = None):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.train()
	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)
		img = img.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)


		out_s = snet(img)
		out_t = tnet(img)
		if opt.aug is None:
			cls_loss = criterionCls(out_s, target)
			if opt.kd_mode in ['logits', 'st']:
				kd_loss = criterionKD(out_s, out_t.detach()) * opt.lambda_kd
			else:
				raise Exception('Invalid kd mode...')
			loss = (1 - opt.lambda_kd) * cls_loss + kd_loss
		else:
			if opt.aug == "diffmix":
				label_arr = target.cpu().numpy()
				label_1, label_2 = torch.LongTensor([map_dict[i][0] for i in label_arr]).cuda(), torch.LongTensor([map_dict[i][1] for i in label_arr]).cuda()
				lam = 0.5
				cls_loss = criterionCls(out_s, label_1) #lam * criterionCls(out_s, label_1) + (1 - lam) * criterionCls(out_s, label_2)
				if opt.kd_mode in ['logits', 'st']:
					kd_loss = criterionKD(out_s, out_t.detach()) * opt.lambda_kd
					loss = (1 - opt.lambda_kd) * cls_loss + kd_loss
				else:
					raise Exception('Invalid kd mode...')
		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % opt.print_freq == 0:
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
	    # here we calculate the 
		logit_t = out_t 
		p = F.softmax(logit_t, dim = -1)
		# p is output probability of logit t
		opt.prob  += [p] # [batch_size, num_classes]
		opt.entropy += [(-p * torch.log(p)).sum(dim = -1)] # [batch_size]
		num = 10
		if len(opt.prob) >= num:
			prob = torch.cat(opt.prob, dim = 0)
			entropy = torch.cat(opt.entropy, dim = 0)
			avg_prob = prob.mean(dim = 0)
			opt.all_avg_prob += [avg_prob]
			opt.prob = [] # reset
			
			all_avg_prob = torch.stack(opt.all_avg_prob, dim = 0) #[num, num_classes]
			avg_prob_std = all_avg_prob.std(dim=0)
			std_str = '%.6f' % avg_prob_std.mean().item()
			if i % opt.print_freq == 0:
				print(f'Check T prob: NumOfSampledStd {len(opt.all_avg_prob)} Epoch {epoch}  MeanStd {std_str} MeanEntropy {entropy.mean().item():.6f}')
	
	return top1.avg, top5.avg, cls_losses.avg, kd_losses.avg, avg_prob_std.mean().item(), entropy.mean().item()


        


def validate_distill(test_loader, nets, criterions, opt):
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.eval()
	end = time.time()
	for _, (img, target) in enumerate(test_loader):
		img = img.cuda()
		target = target.cuda()

		out_s = snet(img)
		out_t = tnet(img)
		cls_loss = criterionCls(out_s, target)
		if opt.kd_mode in ['logits', 'st']:
			kd_loss  = criterionKD(out_s, out_t.detach()) * opt.lambda_kd
		else:
			raise Exception('Invalid kd mode...')

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
	print('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	return top1.avg, top5.avg, cls_losses.avg, kd_losses.avg 



