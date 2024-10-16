'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import numpy as np
import torch
from torch import distributed

import json

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m/s, (1 - m)/s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor



class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # input = input[0]    # TODO: remove this line
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

class BNFeatureHook_ClassStats_Training():
    def __init__(self, module, class_id):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mean = module.class_running_mean.data[class_id]
        self.var = module.class_running_var.data[class_id]

    def hook_fn(self, module, input, output):
        # make sure module has class_running_mean and class_running_var
        input = input[0]
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(self.var - var, 2) + torch.norm(self.mean - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

class BNFeatureHook_ClassStats():
    def __init__(self, module, stats_file, class_id):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.class_id = str(class_id)
        # load json file
        with open(stats_file, 'r') as f:
            file_data = json.load(f)
        self.mean = torch.tensor(file_data[self.class_id]['running_mean'])
        self.var = torch.tensor(file_data[self.class_id]['running_var'])

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)

        # take average of the mean and variance
        stat_mean = torch.mean(self.mean)
        stat_var = torch.mean(self.var)

        # make it the same shape as the mean
        stat_mean = stat_mean.repeat(mean.shape[0]).cuda()
        stat_var = stat_var.repeat(var.shape[0]).cuda()

        r_feature = torch.norm(stat_var - var, 2) + torch.norm(stat_mean - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2

# modified from Alibaba-ImageNet21K/src_files/models/utils/factory.py
def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location="cpu")

    Flag = False
    if "state_dict" in state:
        # resume from a model trained with nn.DataParallel
        state = state["state_dict"]
        Flag = True

    for key in model.state_dict():
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]

        if Flag:
            key = "module." + key

        if key in state:
            ip = state[key]
            # if key in state['state_dict']:
            #     ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print("could not load layer: {}, mismatch shape {} ,{}".format(key, (p.shape), (ip.shape)))
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model