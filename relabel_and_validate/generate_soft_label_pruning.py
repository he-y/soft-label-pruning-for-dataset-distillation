import argparse
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX,
                       RandomHorizontalFlipWithRes,
                       RandomResizedCropWithCoords, mix_aug,
                       load_model_weights, 
                       ComposeWithCoords_Cutout,
                       CutoutPILWithCoords)

from utils import str2bool
import time

import matplotlib.pyplot as plt
import json

import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='FKD Soft Label Generation on ImageNet-1K w/ Mix Augmentation')
parser.add_argument('--dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'imagenet21k', 'tiny'],)
parser.add_argument('--train_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--model', metavar='MODEL', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FKD soft label generation args
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--input-size', default=224, type=int, metavar='S',
                    help='argument in RandomResizedCrop')
parser.add_argument("--min-scale-crops", type=float, default=0.08,
                    help="argument in RandomResizedCrop")
parser.add_argument("--max-scale-crops", type=float, default=1.,
                    help="argument in RandomResizedCrop")
parser.add_argument('--fkd_path', default='./FKD_soft_label',
                    type=str, help='path to save soft labels')
parser.add_argument('--use-fp16', dest='use_fp16', action='store_true',
                    help='save soft labels as `fp16`')
parser.add_argument('--mode', default='fkd_save', type=str, metavar='N',)
parser.add_argument('--fkd-seed', default=42, type=int, metavar='N')

parser.add_argument('--mix-type', default = None, type=str, choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')

# teacher checkpoint
parser.add_argument('--teacher_ckpt', default=None, type=str, help='path to teacher model checkpoint')

# data pruning configs
parser.add_argument('--prune_ratio', type=float, default=0, help='general only (1-prune_ratio) of data')

parser.add_argument('--cfg_yaml', type=str, default=None, help='path to config file')
parser.add_argument('--gpus', type=str, default='0', help='visible devices')

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

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
        cfg_keys = ['basic']
        for cfg_key in cfg_keys:
            for key in cfg['relabel'][cfg_key].keys():
                setattr(args, key, cfg['relabel'][cfg_key][key])
        
        # set store_true args
        for key in cfg['relabel']['store_true']:
            setattr(args, key, True)
    
        # shared config
        if cfg.get('common') is not None:
            common_keys = ['prune', 'basic', 'path']
            for common_key in common_keys:
                if cfg['common'].get(common_key) is None:
                    continue
                for key in cfg['common'][common_key].keys():
                    setattr(args, key, cfg['common'][common_key][key])
        
    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(args.fkd_path):
        os.makedirs(args.fkd_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    print("=> using pytorch pre-trained model '{}'".format(args.model))
    class_dict = {'imagenet1k': 1000, 'imagenet21k':10450, 'tiny': 200}
    num_class = class_dict[args.dataset]


    if args.dataset == 'imagenet1k':
        # model
        model = models.__dict__[args.model](pretrained=True)

        # dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
    elif args.dataset == 'tiny':
        model = models.__dict__[args.model](num_classes=num_class)
        # modifications for tiny imagenet
        # https://github.com/zeyuanyin/tiny-imagenet/tree/main?tab=readme-ov-file
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        assert args.teacher_ckpt is not None, 'teacher checkpoint is not provided'
        model.load_state_dict(torch.load(f'{args.teacher_ckpt}')['model'])

        # model
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                    std=[0.2302, 0.2265, 0.2262])
        
    elif args.dataset == 'imagenet21k':
        import timm
        model = timm.create_model(args.model, pretrained=False, num_classes=10450)

        assert args.teacher_ckpt is not None, 'teacher checkpoint is not provided'
        model = load_model_weights(
            model,
            args.teacher_ckpt,
        )

        normalize = None # donot denormalize since no normalization in training 21k squeezed model

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # freeze all layers
    for name, param in model.named_parameters():
            param.requires_grad = False

    cudnn.benchmark = True

    print("process data from {}".format(args.train_dir))
    if args.dataset == 'imagenet21k':
        normalize = None    # no normalization is used for trianing imagenet21k-P

        train_transform = ComposeWithCoords_Cutout(transforms=[
                RandomResizedCropWithCoords(size=args.input_size,
                                            scale=(args.min_scale_crops,
                                                args.max_scale_crops),
                                            interpolation=InterpolationMode.BILINEAR),
                CutoutPILWithCoords(cutout_factor=0.5),
                transforms.ToTensor(),
            ])
    else:    
        train_transform = ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops,
                                                args.max_scale_crops),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor()])
    
    if normalize is not None:
        # for imagenet1k and tiny
        train_transform.transforms.append(normalize)

    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.train_dir,
        dataset=args.dataset,
        transform=train_transform)

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    drop_last = True if args.prune_ratio > 0 else False # we want to drop the last batch if we are pruning labels
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        drop_last=drop_last,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy)

    sampler_indices_dict = {}
    # compute number of batches per epoch
    args.num_batches = len(train_loader)
    args.total_steps = int(args.epochs * args.num_batches * (1 - args.prune_ratio))

    epochs = args.total_steps // args.num_batches + 1

    batch_counter = 0
    for epoch in tqdm(range(epochs)):
        dir_path = os.path.join(args.fkd_path, 'epoch_{}'.format(epoch))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        batch_counter = save(train_loader, model, dir_path, args, sampler_indices_dict=sampler_indices_dict, epoch=epoch, batch_counter=batch_counter)
        # exit()

    # save the sampler indices
    with open(os.path.join(args.fkd_path, 'sampler_indices_epoch.json'), 'w') as f:
        json.dump(sampler_indices_dict, f)


def save(train_loader, model, dir_path, args, sampler_indices_dict=None, epoch=-1, batch_counter=-1):
    model.eval()
    """Generate soft labels and save"""
    for batch_idx, batch_data in enumerate(train_loader):
        
        if args.dataset == 'imagenet21k':
            images, target, coords_status, coords_cutout, index = batch_data
        else:
            images, target, flip_status, coords_status, index = batch_data

        images = images.cuda()
        images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)
        
        output = model(images)
        if args.use_fp16:
            output = output.half()

        if args.dataset == 'imagenet21k':
            batch_config = [coords_status, coords_cutout, mix_index, mix_lam, mix_bbox, output.cpu()]
        else:
            batch_config = [coords_status, flip_status, mix_index, mix_lam, mix_bbox, output.cpu()]
        batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
        torch.save(batch_config, batch_config_path)
        if sampler_indices_dict is not None:
            if str(epoch) not in sampler_indices_dict:
                sampler_indices_dict.update({str(epoch): []})
            sampler_indices_dict[str(epoch)] += index.tolist()

        # add batch counter
        batch_counter += 1
        
        if batch_counter == args.total_steps:
            break
    
    return batch_counter

if __name__ == '__main__':
    main()
