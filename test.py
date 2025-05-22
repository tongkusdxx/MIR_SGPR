# -*- coding: iso-8859-1 -*-
import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter
import os
import torch

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms

if __name__ == "__main__":
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    opt = TestOptions().parse()
    model = create_model(opt)

    print(f"Current working directory: {os.getcwd()}")

    try:
        model.netEN.module.load_state_dict(
            torch.load("../MIR_SGPR/checkpoints/net_EN.pth")['net'], strict=False)
        model.netDE.module.load_state_dict(
            torch.load("../MIR_SGPR/checkpoints/net_DE.pth")['net'], strict=False)
        model.netMEDFE.module.load_state_dict(
            torch.load("../MIR_SGPR/checkpoints/net_MEDFE.pth")['net'], strict=False)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit(1)

    results_dir = r'./result/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)



    joint_dir=os.path.join(results_dir,'joint')

    for dir_path in [joint_dir]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    mask_paths = sorted(glob('{:s}/*'.format(opt.mask_root)))
    de_paths = sorted(glob('{:s}/*'.format(opt.de_root)))
    st_paths = sorted(glob('{:s}/*'.format(opt.st_root)))


    if not (mask_paths and de_paths and st_paths):
        print("No files found in one or more of the specified directories. Please check the paths.")
        exit(1)

    if len(mask_paths) < len(de_paths):
        mask_paths = (mask_paths * ((len(de_paths) // len(mask_paths)) + 1))[:len(de_paths)]

    image_len = min(len(de_paths), len(st_paths), len(mask_paths))
    # image_len = max(len(de_paths), len(st_paths), len(mask_paths))

    for i in tqdm(range(image_len)):
        path_m = mask_paths[i]
        path_d = de_paths[i]
        path_s = st_paths[i]

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")

        mask_tensor = mask_transform(mask)
        detail_tensor = img_transform(detail)
        structure_tensor = img_transform(structure)

        mask_tensor = torch.unsqueeze(mask_tensor, 0)
        detail_tensor = torch.unsqueeze(detail_tensor, 0)
        structure_tensor = torch.unsqueeze(structure_tensor, 0)

        with torch.no_grad():
            model.set_input(detail_tensor, structure_tensor, mask_tensor)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask_tensor + detail_tensor * (1 - mask_tensor)
            fake_image = (fake_out + 1) / 2.0

        
        original_image = (detail_tensor[0] * 0.5 + 0.5).clamp(0, 1).numpy().transpose((1, 2, 0)) * 255
        original_image = Image.fromarray(original_image.astype(np.uint8))


        detail_image_np = (detail_tensor[0] * 0.5 + 0.5).clamp(0, 1).numpy().transpose((1, 2, 0)) * 255
        detail_image_np = detail_image_np.clip(0, 255).astype(np.uint8)
        mask = mask_tensor[0, 0].cpu().detach().numpy()
        mask = np.expand_dims(mask, axis=-1)

        masked_image_np = detail_image_np * (1 - mask) + (255 * mask).astype(np.uint8)
        masked_image = Image.fromarray(masked_image_np.astype(np.uint8))


        restored_image = (fake_image[0].numpy().transpose((1, 2, 0)) * 255)
        restored_image = Image.fromarray(restored_image.astype(np.uint8))


        total_width = original_image.width + masked_image.width + restored_image.width
        max_height = max(original_image.height, masked_image.height, restored_image.height)

        new_img = Image.new('RGB', (total_width, max_height))

        new_img.paste(original_image, (0, 0))
        new_img.paste(masked_image, (original_image.width, 0))
        new_img.paste(restored_image, (original_image.width + masked_image.width, 0))
        
        
         
        draw = ImageDraw.Draw(new_img)


        save_path = os.path.join(joint_dir, f"combined_{i}.png")
        new_img.save(save_path)