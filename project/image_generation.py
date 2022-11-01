import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import argparse
import random
import numpy as np
import math
import cv2
import streamlit as st

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


def transform_for_visualization(imgs, rows, cols, mean, std):
    # Tile batch images
    c, h, w = imgs[0].shape
    tile = torch.zeros(size=(c, h*rows, w*cols))
    for i in range(rows):
        for j in range(cols):
            index = i*cols + j
            start_h, start_w = h*i, w*j
            tile[:, start_h:start_h+h, start_w:start_w+w] = imgs[index]

    # Visualization
    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    tile = transform(tile)
    tile = tile.cpu().numpy().transpose(1, 2, 0)
    tile = np.clip(tile, 0., 1.) * 255.
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    return tile


# Streamlit Functions
def main(args):
    # for filename in EXTERNAL_DEPENDENCIES.keys():



    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    st.title('Streamlit Random Face Generation')
    st.sidebar.title('Random Seed')
    seed = st.sidebar.slider(label='Random Seed', min_value=0, max_value=10000, value=5000, step=10)

    # Random Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Model
    generator = StyledGenerator().to(device)
    # generator.load_state_dict(torch.load(opt.model_path)['g_running'])
    generator.load_state_dict(torch.load('{}'.format(opt.model_path), map_location=device))
    generator.eval()

    # Mean Styles
    mean_style = get_mean_style(generator, device, style_mean_num=args.style_mean_num)

    # Parameters
    step = int(math.log(opt.img_size, 2)) - 2

    # Generate samples
    imgs = generate_samples(generator, device, n_samples=opt.n_row * opt.n_col, step=step, alpha=opt.alpha,
                            mean_style=mean_style, style_weight=opt.style_weight)
    tile = transform_for_visualization(imgs, rows=opt.n_row, cols=opt.n_col, mean=opt.mean, std=opt.std)

    # Streamlit
    st.title('Streamlit Random Face Generation')
    st.sidebar.title('Random Seed')
    st.image(tile, use_column_width=True)


# Main
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Test StyleGAN')

    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--model_path', default='./pre-trained/Dog(FreezeD).pth', type=str)
    parser.add_argument('--dataset_name', default='Dog', type=str)  # FFHQ, Dog
    parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256

    # Mean Style
    parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
    parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
    parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

    # Sample Generation
    parser.add_argument('--n_row', default=3, type=int)  # For Visualization
    parser.add_argument('--n_col', default=5, type=int)  # For Visualization

    # Style Mixing
    parser.add_argument('--n_source', default=5, type=int)  # cols
    parser.add_argument('--n_target', default=3, type=int)  # rows

    # Transformations
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
    parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

    opt = parser.parse_args()

    main(opt)
