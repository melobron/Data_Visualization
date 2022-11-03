import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import sys
import os
import argparse
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import streamlit as st

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator


# Style GAN Functions
@torch.no_grad()
def get_mean_style(generator, device, style_mean_num):
    mean_style = None

    for _ in range(style_mean_num):
        style = generator.mean_style(torch.randn(1024, 512).to(device))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= style_mean_num
    return mean_style


@torch.no_grad()
def generate_samples(generator, device, n_samples, step, alpha, mean_style, style_weight):
    latent = torch.randn(n_samples, 512).to(device)
    imgs = generator(latent, step=step, alpha=alpha, mean_style=mean_style, style_weight=style_weight)
    imgs = [imgs[i] for i in range(n_samples)]
    return imgs


def transform_for_visualization(imgs, mean, std):
    # Visualization
    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    tile = transform(imgs)
    tile = tile.cpu().numpy().transpose(1, 2, 0)
    tile = np.clip(tile, 0., 1.) * 255.
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    return tile


def make_tile(imgs, rows, cols):
    # Tile batch images
    c, h, w = imgs[0].shape
    tile = torch.zeros(size=(c, h*rows, w*cols))
    for i in range(rows):
        for j in range(cols):
            index = i*cols + j
            start_h, start_w = h*i, w*j
            tile[:, start_h:start_h+h, start_w:start_w+w] = imgs[index]
    return tile


# Streamlit Functions
def image_generation(args, seed):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Model
    generator = StyledGenerator().to(device)
    # generator.load_state_dict(torch.load(opt.model_path)['g_running'])
    # model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}'.format(args.model_name))
    model_path = os.path.join(os.getcwd(), 'pretrained', '{}'.format(args.model_name))
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # Mean Styles
    mean_style = get_mean_style(generator, device, style_mean_num=args.style_mean_num)

    # Parameters
    step = int(math.log(args.img_size, 2)) - 2

    # Generate samples
    imgs = generate_samples(generator, device, n_samples=args.n_row * args.n_col, step=step, alpha=args.alpha,
                            mean_style=mean_style, style_weight=args.style_weight)
    tile = make_tile(imgs, rows=args.n_row, cols=args.n_col)
    tile = transform_for_visualization(tile, mean=args.mean, std=args.std)
    return tile


@torch.no_grad()
def style_mixing(args, seed, n_source, n_target):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Model
    generator = StyledGenerator().to(device)
    # generator.load_state_dict(torch.load(opt.model_path)['g_running'])
    model_path = os.path.join(os.getcwd(), './pretrained', '{}'.format(args.model_name))
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # Mean Styles
    mean_style = get_mean_style(generator, device, style_mean_num=args.style_mean_num)

    # Parameters
    step = int(math.log(args.img_size, 2)) - 2

    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    source_imgs = generator(source_code, step=step, alpha=args.alpha, mean_style=mean_style, style_weight=args.style_weight)
    target_imgs = generator(target_code, step=step, alpha=args.alpha, mean_style=mean_style, style_weight=args.style_weight)

    shape = 4 * (2 ** step)
    imgs = [torch.ones(3, shape, shape).to(device) * -1]

    for i in range(n_source):
        imgs.append(source_imgs[i])

    for i in range(n_target):
        imgs.append(target_imgs[i])
        mixed_imgs = generator([target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
                               step=step, alpha=args.alpha, mean_style=mean_style, style_weight=args.style_weight,
                               mixing_range=(0, 1))
        for j in range(n_source):
            imgs.append(mixed_imgs[j])
    tile = make_tile(imgs, rows=n_target+1, cols=n_source+1)
    tile = transform_for_visualization(tile, mean=args.mean, std=args.std)
    return tile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test StyleGAN')

    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--model_name', default='Dog(FreezeD).pth', type=str)
    parser.add_argument('--dataset_name', default='Dog', type=str)  # FFHQ, Dog
    parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256

    # Mean Style
    parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
    parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
    parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

    # Sample Generation
    parser.add_argument('--n_row', default=3, type=int)  # For Visualization
    parser.add_argument('--n_col', default=5, type=int)  # For Visualization

    # Transformations
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
    parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

    opt = parser.parse_args()

    tile = image_generation(opt, seed=100)
    plt.imshow(tile/255.)
    plt.show()
