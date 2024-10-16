import os

import numpy as np
import torch
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F

import random

# ================== Changes for ImageNet-21K-P =====================
from PIL import ImageDraw

class CutoutPILWithCoords(object):
    """
    modified from: https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/helper_functions/augmentations.py
    """ 
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x, params=None):
        h, w = x.size[1], x.size[0]  # Correcting size unpacking order to (height, width)
        img_draw = ImageDraw.Draw(x)

        if params is None:  # Generate new parameters if none are provided
            h_cutout = int(self.cutout_factor * h + 0.5)
            w_cutout = int(self.cutout_factor * w + 0.5)
            y_c = np.random.randint(h)
            x_c = np.random.randint(w)
            y1 = np.clip(y_c - h_cutout // 2, 0, h)
            y2 = np.clip(y_c + h_cutout // 2, 0, h)
            x1 = np.clip(x_c - w_cutout // 2, 0, w)
            x2 = np.clip(x_c + w_cutout // 2, 0, w)
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            params = torch.FloatTensor([y_c / h, x_c / w, h_cutout / h, w_cutout / w, *fill_color])
        else:  # Use provided parameters to reproduce the cutout
            y_c, x_c, h_cutout, w_cutout, r, g, b = params * torch.FloatTensor([h, w, h, w, 1, 1, 1])
            y1, x1 = np.clip(y_c - h_cutout // 2, 0, h), np.clip(x_c - w_cutout // 2, 0, w)
            y2, x2 = np.clip(y_c + h_cutout // 2, 0, h), np.clip(x_c + w_cutout // 2, 0, w)
            fill_color = (int(r), int(g), int(b))

        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        return x, params

class ComposeWithCoords_Cutout(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords_Cutout, self).__init__(**kwargs)

    def __call__(self, img, coords, coords_cutout):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'CutoutPILWithCoords':
                img, coords_cutout = t(img, coords_cutout)
            else:
                img = t(img)
        return img, coords, coords_cutout
# ================== END: Changes for ImageNet-21K-P =====================

class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return t_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords


class RandomHorizontalFlipWithRes(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return t_F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return t_F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_FKD_info(fkd_path):
    def custom_sort_key(s):
        # Extract numeric part from the string using regular expression
        numeric_part = int(s.split('_')[1].split('.tar')[0])
        return numeric_part

    max_epoch = len(os.listdir(fkd_path))
    batch_list = sorted(os.listdir(os.path.join(
        fkd_path, 'epoch_0')), key=custom_sort_key)
    batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[0]))[1].size()[0]
    last_batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[-1]))[1].size()[0]
    num_img = batch_size * (len(batch_list) - 1) + last_batch_size

    print('======= FKD: dataset info ======')
    print('path: {}'.format(fkd_path))
    print('num img: {}'.format(num_img))
    print('batch size: {}'.format(batch_size))
    # print('max epoch: {}'.format(max_epoch))
    print('================================')
    return max_epoch, batch_size, num_img

# for visualization
def get_class_distribution(indices, ipc):
    classes = [index // ipc for index in indices]

    # intialized with 0
    class_distribution = {key: 0 for key in range(1000)}    # FIX: hard code 1000 for imagenet-1K
    for class_num in classes:
        if type(class_num) == torch.Tensor:
            class_num = class_num.item()    # convert tensor to int
        if class_num in class_distribution:
            class_distribution[class_num] += 1
        else:
            class_distribution[class_num] = 1

    frequencies = [value for key, value in class_distribution.items()]
    
    return frequencies
    
class MultiDatasetImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, mode, dataset='imagenet1k', **kwargs):
        """
        The root should contain the train and val folder
        example: xxx/tiny-imagenet-200/train
        """
        super(MultiDatasetImageFolder, self).__init__(root, **kwargs)

        if mode != 'val':
            return  # only use as val dataset

        if dataset in ['imagenet1k', 'imagenet21k']:
            pass    # keep the original image folder

        elif dataset == 'tiny':
            base_dir = os.path.dirname(root)
            _, self.class_to_idx = MultiDatasetImageFolder.find_tiny_classes(os.path.join(base_dir, 'wnids.txt'))
            self.samples = MultiDatasetImageFolder.make_tiny_dataset(root, self.class_to_idx)
            self.targets = [s[1] for s in self.samples]
            assert len(self.samples) == len(self.targets), "samples and targets should have same length"
            assert len(set(self.targets)) == 200, "tiny imagenet should have 200 classes"
    
    @staticmethod
    def find_tiny_classes(class_file):
        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/tiny_imagenet_dataset.py
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    @staticmethod
    def make_tiny_dataset(root, class_to_idx):
        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/tiny_imagenet_dataset.py
        images = []
        dir_path = root

        dirname = dir_path.split('/')[-1]
        if dirname == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images

class ImageFolder_FKD_MIX(MultiDatasetImageFolder):
    def __init__(self, fkd_path, mode, args_epoch=None, args_bs=None, args_use_batch=False, **kwargs):
        self.fkd_path = fkd_path
        self.mode = mode
        self.use_batch = args_use_batch  # modified to use batch
        super(ImageFolder_FKD_MIX, self).__init__(mode='train', **kwargs)
        self.batch_config = None  # [list(coords), list(flip_status)]
        self.batch_config_idx = 0  # index of processing image in this batch
        self.dataset = "" if 'dataset' not in kwargs else kwargs['dataset'] # use for imagenet21k
        if self.mode == 'fkd_load':
            max_epoch, batch_size, num_img = get_FKD_info(self.fkd_path)
            # if args_epoch > max_epoch:
            #     raise ValueError(f'`--epochs` should be no more than max epoch.')
            if args_bs != batch_size:
                if batch_size == 1000: # special case for ImageNet-1K, IPC=1
                    self.args_bs = batch_size
                else:
                    raise ValueError(f'`--batch-size` should be same in both saving and loading phase. ({args_bs} != {batch_size}) Please use `--gradient-accumulation-steps` to control batch size in model forward phase.')
            # self.img2batch_idx_list = torch.load('/path/to/img2batch_idx_list.tar')
            self.img2batch_idx_list = get_img2batch_idx_list(num_img=num_img, batch_size=batch_size, epochs=max_epoch)
            self.batch_num_per_epoch = len(self.img2batch_idx_list[0])
            self.epoch = None
            self.batch_idx_across_all_epochs = None
            self.batch_list = None
            self.batch_mapping = None
        
    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.mode == 'fkd_save':
            coords_ = None
            flip_ = None
            coords_cutout_ = None # for ImageNet-21K-P
        elif self.mode == 'fkd_load':
            if self.batch_config == None:
                raise ValueError('config is not loaded')
            assert self.batch_config_idx <= len(self.batch_config[0]), "batch config index should be less than length of batch config"

            coords_ = self.batch_config[0][self.batch_config_idx]

            if self.dataset == 'imagenet21k':
                coords_cutout_ = self.batch_config[1][self.batch_config_idx]
            else:
                flip_ = self.batch_config[1][self.batch_config_idx]

            self.batch_config_idx += 1
        else:
            raise ValueError('mode should be fkd_save or fkd_load')

        sample = self.loader(path)

        if self.transform is not None:
            if self.dataset == 'imagenet21k':
                sample_new, coords_status, coords_cutout = self.transform(sample, coords_, coords_cutout_)
                flip_status = None
            else:
                sample_new, flip_status, coords_status = self.transform(sample, coords_, flip_)
        else:
            flip_status = None
            coords_status = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.dataset == 'imagenet21k': # also for fkd_load to prevent None type flip_status for _MapDatasetFetcher
            return sample_new, target, coords_status, coords_cutout, index

        # modifed to return index
        return sample_new, target, flip_status, coords_status, index

    def load_batch_config(self, img_idx):
        """Use the `img_idx` to locate the `batch_idx`

        Args:
            img_idx: index of the first image in this batch
        """
        assert self.epoch != None
        batch_idx = self.img2batch_idx_list[self.epoch][img_idx]
        batch_config_path =  os.path.join(self.fkd_path, 'epoch_{}'.format(self.epoch), 'batch_{}.tar'.format(batch_idx))

        # [coords, flip_status, mix_index, mix_lam, mix_bbox, soft_label]
        config = torch.load(batch_config_path)
        self.batch_config_idx = 0
        self.batch_config = config[:2]
        return config[2:]

    def load_batch_config_by_batch_idx(self, img_idx):
        """
        Modified to load batch config by batch_idx

        self.batch_idx_across_all_epochs: the index of the batch across all epochs
            - should be set for every batch
        """
        if self.batch_list is not None:
            assert self.batch_mapping is not None
            current_batch_idx = self.batch_mapping[img_idx]
        else:
            assert self.batch_idx_across_all_epochs != None
            current_batch_idx = self.batch_idx_across_all_epochs

        # dynamically compute the epoch
        epoch = current_batch_idx // self.batch_num_per_epoch
        # compute relative batch index
        batch_idx = current_batch_idx % self.batch_num_per_epoch

        # below are SAME as load_batch_config
        batch_config_path =  os.path.join(self.fkd_path, 'epoch_{}'.format(epoch), 'batch_{}.tar'.format(batch_idx))
        # print(f"load batch config from idx {current_batch_idx}:\n{batch_config_path}")

        # [coords, flip_status, mix_index, mix_lam, mix_bbox, soft_label]
        config = torch.load(batch_config_path)
        self.batch_config_idx = 0
        self.batch_config = config[:2]
        return config[2:]
        
    def set_batch(self, batch_idx):
        self.batch_idx_across_all_epochs = batch_idx

    def set_batch_list(self, batch_list, mapping):
        self.batch_list = batch_list
        self.batch_mapping = mapping

    def set_epoch(self, epoch):
        self.epoch = epoch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.cutmix, args.cutmix)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None and bbox is not None
        rand_index = rand_index.cuda()
        lam = lam
        bbx1, bby1, bbx2, bby2 = bbox
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.mixup, args.mixup)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None
        rand_index = rand_index.cuda()
        lam = lam
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None

def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == 'mixup':
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == 'cutmix':
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None

def get_img2batch_idx_list(num_img = 50000, batch_size = 1024, seed=42, epochs=300):
    train_dataset = torch.utils.data.TensorDataset(torch.arange(num_img))
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    img2batch_idx_list = []
    for epoch in range(epochs):
        img2batch_idx = {}
        for batch_idx, img_indices in enumerate(batch_sampler):
            img2batch_idx[img_indices[0]] = batch_idx

        img2batch_idx_list.append(img2batch_idx)
    return img2batch_idx_list

# for imagenet-21k
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
