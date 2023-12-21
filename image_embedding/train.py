import os
import cv2
import math
import time
import pickle
import random
import argparse
import albumentations
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from logger import create_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.distributed as dist
import apex
from apex import amp
from apex.parallel import DistributedDataParallel
from utils import *
from dataset import ImageEmbeddingDataset_general
from build_model import clip_vit
from config_clip import get_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--arc_s', type=float, default=45.0)
    parser.add_argument('--arc_m', type=float, default=0.3)
    parser.add_argument('--n_batch_log', type=int, default=100)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    config = get_config()
    return args, config


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(c_epoch, model, loader, optimizer, criterion_class):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    bar = tqdm(loader)
    lr = get_train_epoch_lr(c_epoch, args.n_epochs, args.init_lr)
    set_lr(optimizer, lr)
    warm_up_step = 0
    for (imgs, gts) in bar:
        if c_epoch<=args.warm_up_epochs:
            lr = get_warm_up_lr(args.warm_up_epochs, c_epoch, warm_up_step, args.init_lr, len(bar))
            set_lr(optimizer, lr)
            warm_up_step += 1
        imgs, gts = imgs.cuda(), gts.cuda()
        preds = model(imgs)
        loss = criterion_class(preds, gts)
        optimizer.zero_grad(set_to_none=True)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        reduced_loss = reduce_tensor(loss.data)
        losses.update(to_python_float(reduced_loss), imgs.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        bar.set_description('Loss_cur: %.5f, Loss_avg: %.5f' % (losses.val, losses.avg))
        if batch_time.count%args.n_batch_log == 0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('Epoch: %d | Iter: [%d/%d], lr: %.7f, Memory_used: %.0fMB, loss_cur: %.5f, loss_avg: %.5f, batch_time_avg: %.3f, time_total: %.3f' % (c_epoch, batch_time.count, len(loader), lr, mu, losses.val, losses.avg, batch_time.avg, batch_time.sum))
    return losses.avg

def val_epoch(model, valid_loader, criterion_class):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    end = time.time()
    bar = tqdm(valid_loader)
    model.eval()
    with torch.no_grad():
        for (imgs, gts) in bar:
            imgs, gts = imgs.cuda(), gts.cuda()
            preds = model(imgs)
            top1, top5 = accuracy(preds.data, gts, topk=(1, 5))
            loss = criterion_class(preds, gts)
            
            reduced_loss = reduce_tensor(loss.data)
            top1 = reduce_tensor(top1)
            top5 = reduce_tensor(top5)
            losses.update(to_python_float(reduced_loss), imgs.size(0))
            top1_acc.update(to_python_float(top1), imgs.size(0))
            top5_acc.update(to_python_float(top5), imgs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_time.count%args.n_batch_log == 0:
                logger.info('Valid || loss_avg: %.5f, batch_time_sum: %.3f, top1_acc: %.2f, top5_acc: %.2f' % 
                                (losses.avg, batch_time.sum, top1_acc.avg, top5_acc.avg))
        return losses.avg, top1_acc.avg


def main(config):
    df = pd.read_csv(args.csv_dir)
    # get train and valid dataset
    dataset_train = ImageEmbeddingDataset_general(df, args.fold, 'train', args.image_size)
    dataset_valid = ImageEmbeddingDataset_general(df, args.fold, 'valid', args.image_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, sampler=valid_sampler,
                                               drop_last=True)
    
    # model
    model = clip_vit(config, logger)
    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=config.Optimizer.weight_decay)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = DistributedDataParallel(model, delay_allreduce=True)
    # loss func
    criterion_class = ArcFaceLoss(s=args.arc_s, m=args.arc_m).cuda()
    if config.MODEL.finetune != None:
        load_checkpoint(config, model, logger)
    # strat training
    logger.info("Start training")
    start_time = time.time()
    best_top1_acc = -1
    args.n_epochs += 1
    for epoch in range(1, args.n_epochs):
        logger.info(f"----------[Epoch {epoch}]----------")   
        
        # dataset shuffle
        train_loader.sampler.set_epoch(epoch)
        # train & valid
        t_loss = train_epoch(epoch, model, train_loader, optimizer, criterion_class)
        v_loss, top1_acc = val_epoch(model, valid_loader, criterion_class)
        logger.info(f"Epoch: {epoch} || Train_Loss: {t_loss:.5f}, Val_Loss: {v_loss:.5f}, top1_acc: {top1_acc:.2f}%")
        
        # save model
        if args.local_rank == 0:
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.NAME}_best.pth')
                save_checkpoint(model, save_path, config=config)
                logger.info(f"Save best model to {save_path}, with best acc_1: {best_top1_acc}")
        logger.info(f'Epoch {epoch} spent time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
        
    # finish training
    save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.NAME}_last.pth') 
    save_checkpoint(model, save_path, config=config)
    logger.info(f"save last model to: {save_path}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time in total: {}'.format(total_time_str))


if __name__ == '__main__':
    args, config = parse_args()
    config.defrost()
    config.IMG_SIZE = args.image_size
    config.freeze()
    
    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    args.world_size = dist.get_world_size()
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.NAME}")
    logger.info(config.dump())
    main(config)

