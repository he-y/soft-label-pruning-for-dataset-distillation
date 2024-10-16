import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
from torchvision import transforms
from utils_tiny import *
from utils_recover import BNFeatureHook, BNFeatureHook_ClassStats, BNFeatureHook_ClassStats_Training

import sys
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet_class import resnet18_class, resnet50_class

from PIL import Image
from tqdm import trange

def get_images(args, model_teacher, hook_for_display, ipc, class_id):
    print("generating one IPC images (200)")
    save_every = 100
    batch_size = args.batch_size

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            if args.bn_hook_type == 'class':
                loss_r_feature_layers.append(BNFeatureHook(module))
            elif args.bn_hook_type == 'class_stats':
                # FIXME: stats_file is None
                loss_r_feature_layers.append(BNFeatureHook_ClassStats(module, stats_file=None, class_id=class_id))
            elif args.bn_hook_type == 'class_stats_training':
                loss_r_feature_layers.append(BNFeatureHook_ClassStats_Training(module, class_id))
            else:
                raise ValueError("unknown bn_hook_type")

    for kk in range(0, ipc, batch_size):

        size = min(batch_size, ipc)
        targets = [class_id]*size
        targets = torch.LongTensor(targets).to('cuda')

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, 64, 64), requires_grad=True, device='cuda',
                             dtype=data_type)

        iterations_per_layer = args.iteration

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in trange(iterations_per_layer, desc='ITER'):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            # FIXME: potential bug here: the augmentation is same for all images in the batch
            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(-args.jitter, args.jitter)
            off2 = random.randint(-args.jitter, args.jitter)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
            # inputs_jit (batch_size, 3, 64, 64)

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)
            # outputs (batch_size, 200)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
            # rescale = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # final loss
            loss = loss_ce + args.r_bn * loss_r_bn_feature

            if iteration % save_every == 0 and args.verifier:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip_tiny(inputs.data)

        if args.store_last_images:
            save_inputs = inputs.data.clone()  # using multicrop, save the last one
            save_inputs = denormalize_tiny(save_inputs)
            custom_save_images(args, save_inputs, targets)

def custom_save_images(args, images, targets):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main_syn(class_id, args):
    model_teacher = None
    if args.bn_hook_type == 'class_stats_training':
        if args.arch_name == 'resnet18':
            model_teacher = resnet18_class(num_classes=200)
        elif args.arch_name == 'resnet50':
            model_teacher = resnet50_class(num_classes=200)
        else:
            raise NotImplementedError
    else:
        model_teacher = models.__dict__[args.arch_name](num_classes=200)
    model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher.maxpool = nn.Identity()
    checkpoint = torch.load(args.arch_path, map_location="cpu")
    model_teacher.load_state_dict(checkpoint["model"])

    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if args.verifier:
        model_verifier = models.__dict__[args.verifier_arch](num_classes=200)
        model_verifier.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model_verifier.maxpool = nn.Identity()
        checkpoint = torch.load(args.verifier_arch_path, map_location="cpu")
        model_verifier.load_state_dict(checkpoint["model"])

        model_verifier = model_verifier.cuda()
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False
        hook_for_display = lambda x, y: validate(x, y, model_verifier)
    else:
        hook_for_display = None

    get_images(args, model_teacher, hook_for_display, args.ipc, class_id)
    del model_teacher
    if args.verifier:
        del model_verifier


def parse_args():
    parser = argparse.ArgumentParser("SRe2L: recover data from pre-trained model")
    """ Data save flags """
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-last-images', action='store_true',
                        help='whether to store best images')
    """ Optimization related flags """
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    """ Model related flags """
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--arch-path', type=str, default='')
    parser.add_argument('--verifier-arch-path', type=str, default='')
    """ Training Helpers """
    parser.add_argument('--ipc', default=10, type=int)
    parser.add_argument('--ipc-start', default=0, type=int)
    parser.add_argument('--ipc-end', default=50, type=int)
    parser.add_argument('--bn-hook-type', type=str, default='class_stats', choices=['class', 'class_stats', 'class_stats_training'], help='type of bn hook')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path, exist_ok=True)

    for c in range(args.ipc_start, args.ipc_end):
        print(f'class = {c}')
        main_syn(c, args)
