"""
code adapted from: https://github.com/VILA-Lab/SRe2L/blob/main/SRe2L/validate/train_FKD.py
"""
import argparse
import math
import os
import shutil
import sys
import time
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils import AverageMeter, accuracy, get_parameters, str2bool
import utils_pruning

from aim import Run, Figure
from aim_utils.aim_logger import aim_log, aim_hyperparam_log, aim_terminal_log

from torch.utils.data import Sampler

# import px
import plotly.express as px
import matplotlib.pyplot as plt

sys.path.append('..')
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher

from utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX, MultiDatasetImageFolder,
                               RandomHorizontalFlipWithRes,
                               RandomResizedCropWithCoords, mix_aug, 
                               ComposeWithCoords_Cutout,
                               CutoutPILWithCoords,
                               get_class_distribution)

from tqdm import trange

from ema_pytorch import EMA

import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'imagenet21k', 'tiny'],)
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train_dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output_dir', type=str,
                        default='./save/1024', help='path to output dir')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd_path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--mix_type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--val_interval', default=10, type=int, help='validation interval')
    
    # data pruning configs
    parser.add_argument('--prune_label', type=bool, default=True, help='prune label')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='prune away `prune_ratio` of data')
    parser.add_argument('--prune_metric', type=str, default='forgetting', choices=['random', 'order', 'aum', 'forgetting', 'yoco', 'entropy', 'el2n'], help='prune method')

    parser.add_argument('--granularity', type=str, default='epoch', choices=['epoch', 'batch', 'batch_to_epoch'], help='granularity of pruning')
    parser.add_argument('--smoothing_lr', type=str2bool, default=False, help='use smoothing lr scheduler')
    parser.add_argument('--smoothing_lr_strength', type=float, default=2, help='strength of smoothing lr scheduler')
    parser.add_argument('--ema', type=str2bool, default=False, help='whether to use ema model to validate')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema decay rate')

    parser.add_argument("--min-scale-crops", type=float, default=0.08,
                        help="argument in RandomResizedCrop")
    # imagent21k-P configs
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing')
    
    # aim logger
    parser.add_argument('--use_aim', default=True, action='store_false', help='use aim logger')
    parser.add_argument('--run_name', type=str, default="default", help='name of the run')
    parser.add_argument('--exp_name', type=str, default="label_pruning", help='name of the experiment')
    parser.add_argument('--tag_name', type=str, default=None, help='tag of the run')
    parser.add_argument('--cfg_yaml', type=str, default=None, help='path to config file')
    
    parser.add_argument('--gpus', type=str, default='0', help='visible devices')

    parser.add_argument('--train_epochs', type=int, default=-1, help='specifying the number of training epochs')

    args = parser.parse_args()
    
    if args.cfg_yaml:
        # load yaml config
        import yaml

        def load_config(config_file):
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config

        # Load the YAML configuration, for example:
        cfg = load_config(args.cfg_yaml)

        # set key-value
        cfg_keys = ['basic', 'path', 'aim']
        for cfg_key in cfg_keys:
            for key in cfg['validate'][cfg_key].keys():
                setattr(args, key, cfg['validate'][cfg_key][key])
        
        # set store_true args
        for key in cfg['validate']['store_true']:
            setattr(args, key, True)
        
        # shared config
        # check if common config exists
        if cfg.get('common') is not None:
            common_keys = ['prune', 'basic', 'path']
            for common_key in common_keys:
                if cfg['common'].get(common_key) is None:
                    continue
                for key in cfg['common'][common_key].keys():
                    setattr(args, key, cfg['common'][common_key][key])

        args.output_dir = args.output_dir + f'_T[{int(args.temperature)}]'
        args.run_name = args.run_name + f'_T[{int(args.temperature)}]'

        # add prune config taggings
        if args.prune_label:
            suffix = f'_M[{args.prune_metric}]'

            if 'batch' in args.granularity:
                suffix += f'_BATCH'

            suffix += f'_R[{args.prune_ratio}]'
        
            if args.train_epochs > 0:
                suffix += f'_EP[{args.train_epochs}]'
            
            if args.adamw_lr != 0.001:
                suffix += f'_LR[{args.adamw_lr}]'

            args.output_dir = args.output_dir + suffix
            args.run_name = args.run_name + suffix
    

    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    args.mode = 'fkd_load'
    return args

def load_dataset(args):
    # Data loading
    if args.dataset == 'imagenet1k':
        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_transform = ComposeWithCoords(transforms=[
                    RandomResizedCropWithCoords(size=image_size,
                                                scale=(args.min_scale_crops, 1),
                                                interpolation=InterpolationMode.BILINEAR),
                    RandomHorizontalFlipWithRes(),
                    transforms.ToTensor(),
                    normalize,
                ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset == 'tiny':
        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/train.py
        image_size = 64
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                    std=[0.2302, 0.2265, 0.2262])
        train_transform = ComposeWithCoords(transforms=[
                    RandomResizedCropWithCoords(size=image_size,
                                                scale=(args.min_scale_crops, 1),
                                                interpolation=InterpolationMode.BILINEAR),
                    RandomHorizontalFlipWithRes(),
                    transforms.ToTensor(),
                    normalize,
                ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    elif args.dataset == 'imagenet21k':
        image_size = 224
        normalize = None    # no normalization is used for trianing imagenet21k-P

        train_transform = ComposeWithCoords_Cutout(transforms=[
                RandomResizedCropWithCoords(size=image_size,
                                            scale=(0.08, 1),
                                            interpolation=InterpolationMode.BILINEAR),
                CutoutPILWithCoords(cutout_factor=0.5),
                transforms.ToTensor(),
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        dataset=args.dataset,   # use to distinguish imagenet21k
        transform=train_transform)

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)

    if args.prune_label is False:
        """Normal Setup"""
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=args.workers, pin_memory=True)
    else:
        """
        Epoch Sampler Setup
        Ensure that the same data is sampled for a given epoch
        """
        sampler = utils_pruning.EpochBatchSampler(train_dataset, generator=generator)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=args.workers, pin_memory=True)
        
        # check if the sampler indices are already generated
        if os.path.exists(os.path.join(args.fkd_path, 'sampler_indices_epoch.json')):
            with open(os.path.join(args.fkd_path, 'sampler_indices_epoch.json'), 'r') as f:
                sampler.indices_epoch = json.load(f)
                print(f"Loaded sampler indices from {args.fkd_path}")
            # convert all keys to int
            sampler.indices_epoch = {int(k): v for k, v in sampler.indices_epoch.items()}
        else:
            """Preparing Sampling indices for each epoch."""
            for epoch in trange(args.start_epoch, args.epochs, desc="Preparing Sampling Indices"):
                sampler.set_epoch(epoch)
                train_loader.dataset.set_epoch(epoch)
                for batch_idx, batch_data in enumerate(train_loader):
                    # simply iterate through the dataset to generate indices
                    continue
        
            # save the sampler.indices_epoch to a file
            with open(os.path.join(args.fkd_path, 'sampler_indices_epoch.json'), 'w') as f:
                json.dump(sampler.indices_epoch, f)

        if 'batch' in args.granularity:
            train_loader.dataset.use_batch = True
            sampler.use_batch(batch_size=args.batch_size)
    
    # load validation data
    val_dataset = MultiDatasetImageFolder(mode='val', dataset=args.dataset, root=args.val_dir, transform=val_transform)
    val_bs = int(args.batch_size/4) if args.batch_size > 64 else 64 # ensure val_bs is not too small
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')

    return train_dataset, train_loader, val_loader

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # get training dynamics
    TD_logger = None

    os.makedirs(args.output_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.output_dir)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    global run
    if args.use_aim:
        run_hash=None
        run = Run(experiment=args.exp_name, repo='validate_result', run_hash=run_hash, system_tracking_interval=None)
        run.name = args.run_name
        aim_hyperparam_log(run, args)
    else:
        run = None

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _, train_loader, val_loader = load_dataset(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))
    # imagenet21k-P is the pruned version of imagenet21k, containing 10,450 classes
    class_dict = {'imagenet1k': 1000, 'imagenet21k': 10450, 'tiny': 200}
    num_class = class_dict[args.dataset]
    model = torchvision.models.__dict__[args.model](weights=None, num_classes=num_class)
    if args.dataset == 'tiny':
        # modifications for tiny imagenet
        # https://github.com/zeyuanyin/tiny-imagenet/tree/main?tab=readme-ov-file
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()

    if args.train_epochs > 0:
        # scaling =  args.train_epochs / args.epochs 
        # args.adamw_lr = args.adamw_lr * scaling
        print(f"Scaling the learning rate to {args.adamw_lr}")
    else:
        args.train_epochs = args.epochs

    if args.dataset == 'tiny':
        model = model.cuda()    # a slight performance degradation is observed when using DataParallel
    else:
        model = nn.DataParallel(model).cuda()
    model.train()

    if args.ema:
        ema_model = EMA(model, beta=args.ema_decay)
    else:
        ema_model = None

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.train_epochs)) if step <= args.train_epochs else 0, last_epoch=-1)

        # handle special cases
        if args.smoothing_lr:
            # smoothing with strength 2
            scheduler = LambdaLR(optimizer,
                                lambda step: 0.5 * (1. + math.cos(math.pi * step / args.train_epochs / args.smoothing_lr_strength)) if step <= args.train_epochs else 0, last_epoch=-1)

        if args.sgd and (args.dataset == 'tiny'):
            # lr warm up for tiny-imagenet
            args.lr_warmup_epochs = 5
            args.lr_warmup_decay = 0.01
            args.lr_warmup_method = 'linear'
            args.lr_min = 0.0
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
    else:
        # default for SRe2L
        scheduler = LambdaLR(optimizer,
                            lambda step: (1.0-step/args.train_epochs) if step <= args.train_epochs else 0, last_epoch=-1)

    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    total_steps = -1
    if args.granularity == 'batch':
        print("Granulairty: batch")
        granularity = 'batch'
        total_steps = args.train_epochs * args.train_loader.dataset.batch_num_per_epoch
        val_interval = args.val_interval*args.train_loader.dataset.batch_num_per_epoch
        save_interval = args.train_loader.dataset.batch_num_per_epoch
        num_steps = 1
    elif args.granularity == 'batch_to_epoch':
        print("Granulairty: batches to form epoch")
        granularity = 'batch_to_epoch'
        total_steps = args.train_epochs * args.train_loader.dataset.batch_num_per_epoch
        val_interval = args.val_interval*args.train_loader.dataset.batch_num_per_epoch
        save_interval = 1
        num_steps = args.train_loader.dataset.batch_num_per_epoch
    else:   # train use epoch
        print("Granulairty: epoch")
        granularity = 'epoch'
        total_steps = args.train_epochs
        val_interval = args.val_interval
        save_interval = 1
        num_steps = 1

    # === create pool for pruning ===
    pool = list(range(total_steps))
        
    assert int((1-args.prune_ratio)*total_steps) <= len(pool), f"Not enough pool size, {int((1-args.prune_ratio)*total_steps)} > {len(pool)}"
    pool = pool[:int((1-args.prune_ratio)*total_steps)]
    print(f"Generated pool of size {len(pool)}")
    print(f"Pool: {pool[:10]}")
    # ================================

    tracker = []
    stat_tracker = {}
    print(f"Total Steps: {total_steps}")
    for step in trange(0, total_steps, num_steps, desc="Training"):
        if step % save_interval == 0:
            args.objs = AverageMeter()
            args.top1 = AverageMeter()
            args.top5 = AverageMeter()

        global logging_metrics
        logging_metrics = {}

        abs_step = step
        step_list = None

        # randomly sample epoch from pool
        if args.prune_metric == 'random':
            if granularity in ['epoch', 'batch']:
                random_step = torch.randint(0, len(pool), (1,)).item()
                step = pool[random_step]
                tracker.append(step)
            else:
                """
                speical case for batch_to_epoch, which pass a list of batch indices
                """
                step_list = []
                for i in range(args.train_loader.dataset.batch_num_per_epoch):
                    random_step = torch.randint(0, len(pool), (1,)).item()
                    step_list.append(pool[random_step])
                    
                # drop last batch
                for i, batch_idx in enumerate(step_list):
                    if (batch_idx + 1) % args.train_loader.dataset.batch_num_per_epoch == 0:
                        step_list[i] = batch_idx - 1

                tracker += step_list
                
        elif args.prune_metric == 'order':
            if granularity in ['epoch', 'batch']:
                step = step
                tracker.append(step)
            else:
                step_list = list(range(step, step+args.train_loader.dataset.batch_num_per_epoch))
                # drop last batch
                for i, batch_idx in enumerate(step_list):
                    if (batch_idx + 1) % args.train_loader.dataset.batch_num_per_epoch == 0:
                        step_list[i] = batch_idx - 1
        else:
            raise NotImplementedError

        if granularity == 'epoch':
            current_epoch = abs_step+1
        else:
            current_epoch = abs_step//args.train_loader.dataset.batch_num_per_epoch+1

        train(model, args, step_list if step_list else step,
              TD_logger=TD_logger, abs_step=abs_step, interval=save_interval, ema_model=ema_model)


        if (abs_step % val_interval == 0) or (abs_step == total_steps - num_steps):
            top1 = validate(model if ema_model is None else ema_model, args, current_epoch)
        else:
            top1 = 0

        aim_log(run, logging_metrics, current_epoch+1) 

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        if (abs_step % save_interval) == 0 or (abs_step == total_steps - num_steps):
            save_checkpoint({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': args.best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, is_best, output_dir=args.output_dir)

    if run: # aim save terminal log
        # track the histogram of epoch tracker
        run.track(Figure(px.histogram(x=tracker, nbins=100)), name=f'{granularity}_tracker', step=0, context={'metric': f'{granularity}_tracker'})
        dir_name = args.output_dir
        aim_terminal_log(run, dir_name, args)

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, step, TD_logger=None, abs_step=None, interval=1, ema_model=None):
    objs = args.objs
    top1 = args.top1
    top5 = args.top5

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    if args.dataset == 'tiny':
        loss_function_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        print("=>Using KLDivLoss with log_target=True")

    model.train()
    if args.granularity == 'batch_to_epoch':
        "set batch for sampler and dataset"
        if args.prune_label:
            args.train_loader.sampler.set_batch_list(step)
            mappings = args.train_loader.sampler.get_batch_list_img_mapping()
        args.train_loader.dataset.set_batch_list(step, mappings)
    elif args.granularity == 'batch':
        if args.prune_label:
            args.train_loader.sampler.set_batch(step)
        args.train_loader.dataset.set_batch(step)
    elif args.granularity == 'epoch':
        "set epoch for sampler and dataset"
        if args.prune_label:
            args.train_loader.sampler.set_epoch(step)  # set epoch for sampler
        args.train_loader.dataset.set_epoch(step)

        print("Epoch set to", step)
        # print(args.train_loader.sampler.indices_epoch[step][:10])
    else:
        raise NotImplementedError
    

    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status, idx = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]   # additional data from _MapDatasetFetcher
        

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        all_outputs = []  # Initialize a list to store all outputs
        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            if args.label_smoothing > 0:
                num_classes = partial_soft_label.size(1)  # batch_size x num_classes
                uniform_distribution = torch.full((args.batch_size, num_classes), 1 / num_classes, device=partial_soft_label.device)

                # Adjust partial_soft_label with label smoothing
                partial_soft_label = (1 - args.label_smoothing) * partial_soft_label + args.label_smoothing * uniform_distribution

            output = model(partial_images)
            all_outputs.append(output.detach().cpu().type(torch.half)) # log all outputs
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            if args.dataset == 'tiny':
                partial_soft_label = F.log_softmax(partial_soft_label/args.temperature, dim=1)
            else:
                partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
            output = F.log_softmax(output/args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)

            if args.dataset == 'tiny':
                assert accum_step == 1, "accum_step should be 1 for tiny-imagenet"
                # scale the loss by the temperature for tiny-imagenet
                loss = loss * (args.temperature**2)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            
            if ema_model is not None:
                ema_model.update()
        
        all_outputs = torch.cat(all_outputs, dim=0)

        optimizer.step()

    if (abs_step+1) % interval == 0:
        metrics = {
            "train/loss": objs.avg,
            "train/Top1": top1.avg,
            "train/Top5": top5.avg,
            "train/lr": scheduler.get_last_lr()[0],
            }
            # "train/epoch": epoch,}
        logging_metrics.update(metrics)

        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(abs_step, scheduler.get_last_lr()[0], objs.avg) + \
                    'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                    'Top-5 err = {:.6f},\t'.format(100 - top5.avg)
        print(printInfo)
    # breakpoint()

def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        # 'val/epoch': epoch,
    }

    logging_metrics.update(metrics)

    return top1.avg

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)



if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    main()
