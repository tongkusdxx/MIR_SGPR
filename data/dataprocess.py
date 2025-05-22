# -*- coding: gbk -*-
import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import cv2



def resize(img, size, center_crop=True):
    h, w = img.shape[:2]
    size = tuple(size)

    if center_crop:
        diffh = (h - size[0]) // 2
        diffw = (w - size[1]) // 2
        img = img[diffh:diffh + size[0], diffw:diffw + size[1]]
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def to_tensor(pic, norm=False, is_mask=False):
    """
    将输入图像或掩膜转换为 PyTorch 的 Tensor 格式。
    """

    if isinstance(pic, torch.Tensor):
        return pic


    if not isinstance(pic, np.ndarray):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))


    if len(pic.shape) == 2:
        if is_mask:
            pic = pic[:, :, None]
        else:
            pic = np.stack([pic] * 3, axis=-1)

    elif len(pic.shape) == 3 and pic.shape[2] == 3:
        pass
    else:
        raise ValueError('pic should have 3 channels but got shape {}'.format(pic.shape))


    pic = torch.from_numpy(pic.transpose((2, 0, 1)))

    if norm:
        if pic.dtype == torch.uint8:
            pic = pic.float().div(255)

    return pic


def load_masked_position_encoding(mask):
    """
    生成掩膜的相对位置编码、绝对位置编码和方向信息。
    """
    ones_filter = np.ones((3, 3), dtype=np.float32)
    d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    str_size = 256
    pos_num = 128

    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = 1. - (mask / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        for d, d_filter in enumerate([d_filter1, d_filter2, d_filter3, d_filter4]):
            m = cv2.filter2D(mask3, -1, d_filter)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, d] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (str_size / 2)
    rel_pos = (rel_pos * pos_num).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, pos_num - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct


def load_image(img_path, mask_path, sigma256=3.0, input_size=256):
    """
    加载图像并处理掩膜。
    """
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img = resize(img, (input_size, input_size), center_crop=True)
    img_256 = resize(img, (256, 256))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    mask_256 = resize(mask, (256, 256))

    gray_256 = rgb2gray(img_256)
    edge_256 = cv2.Canny(gray_256.astype(np.uint8), sigma256, sigma256).astype(float)

    img_512 = resize(img, (512, 512))
    rel_pos, abs_pos, direct = load_masked_position_encoding(mask)

    batch = {
        'image': to_tensor(img),
        'img_256': to_tensor(img_256, norm=True),
        'mask': to_tensor(mask, is_mask=True),
        'mask_256': to_tensor(mask_256, is_mask=True),
        'edge_256': to_tensor(edge_256),
        'img_512': to_tensor(img_512),
        'rel_pos': torch.LongTensor(rel_pos),
        'abs_pos': torch.LongTensor(abs_pos),
        'direct': torch.LongTensor(direct),
    }
    return batch


class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, mask_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize)),
            transforms.ToTensor()
        ])
        self.Train = train
        self.opt = opt


        self.de_paths = sorted(glob(f'{de_root}/*'))
        self.st_paths = sorted(glob(f'{st_root}/*'))
        self.mask_paths = sorted(glob(f'{mask_root}/*'))


        self.N_mask = len(self.mask_paths)

        
    def __getitem__(self, index):
        de_img_path = self.de_paths[index]
        st_img_path = self.st_paths[index]
        mask_img_path = self.mask_paths[random.randint(0, self.N_mask - 1)]


        batch = load_image(de_img_path, mask_img_path, sigma256=self.opt.sigma256, input_size=self.opt.fineSize)

        de_img = self.img_transform(Image.open(de_img_path).convert('RGB'))
        st_img = self.img_transform(Image.open(st_img_path).convert('RGB'))

        return {
            'de_img': de_img,
            'st_img': st_img,
            **batch
        }

    def __len__(self):
        return len(self.de_paths)
