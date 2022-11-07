import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim

import sys
from glob import glob
import os
import argparse
import random
import numpy as np
import math
import cv2
import streamlit as st

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator
from algorithms.gan_inversion import *


############################## Arguments ##############################
parser = argparse.ArgumentParser(description='Invert StyleGAN')

parser.add_argument('--exp_detail', type=str, default='Invert StyleGAN')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Inverting
parser.add_argument('--latent_type', type=str, default='mean_style')
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)

# Mean Style
parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()


############################## Functions ##############################
def run(inverter, img_path):
    img = inverter.read_img(img_path=img_path).to(inverter.device)
    img_numpy = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    latent = inverter.initial_latent(latent_type=inverter.latent_type).to(inverter.device)
    latent.requires_grad = True
    optimizer = optim.SGD([latent], lr=inverter.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2, eta_min=1e-5, last_epoch=-1)

    col1, col2 = st.columns(2)
    with col1:
        target_img = st.empty()
        target_img.image(img_numpy/255., use_column_width=True, caption='Target Image')
    with col2:
        image_location = st.empty()
    progress_bar = st.empty()

    for iteration in range(1, inverter.iterations + 1):
        decoded_img = inverter.G.forward_from_style(style=latent, step=inverter.step, alpha=inverter.alpha,
                                                mean_style=inverter.mean_style, style_weight=inverter.style_weight)
        lpips_loss = inverter.lpips_criterion(decoded_img, img)
        mse_loss = inverter.MSE_criterion(decoded_img, img)
        loss = lpips_loss + mse_loss
        loss.backward()
        optimizer.step()

        # print('Iteration {} | total loss:{} | lpips loss:{}, mse loss:{}'.format(
        #     iteration, loss.item(), lpips_loss.item(), mse_loss.item()
        # ))

        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-m / s for m, s in zip(inverter.mean, inverter.std)], std=[1 / s for s in inverter.std])
        ])
        sample = torch.squeeze(decoded_img, dim=0)
        sample = reverse_transform(sample)
        sample = sample.detach().cpu().numpy().transpose(1, 2, 0)
        sample = np.clip(sample, 0., 1.) * 255.
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        image_location.image(sample/255., use_column_width=True, caption='Prediction')
        progress_bar.progress(iteration / inverter.iterations)

        scheduler.step()

############################## Streamlit ##############################
if __name__ == '__main__':
    st.title('GAN Inversion')
    st.sidebar.title('Choose Variables')

    # Domain Select Box
    domain = st.sidebar.selectbox(label='Select Domain',
                                  options=['Dog', 'Cat', 'AFAD'])

    # Sample Image Selection
    img_dir = os.path.join(os.getcwd(), 'sample_imgs/{}'.format(domain))
    img_paths = make_dataset(img_dir)
    sample_img_name = st.sidebar.selectbox(label='Select Image', options=[i for i in range(1, len(img_paths))])

    # # Start Inverter button
    # col1, col2 = st.columns(2)
    # with col1:
    #     start_button = st.empty()
    #     start = start_button.button('Start', disabled=False, key='1')
    # with col2:
    #     reset_button = st.empty()
    #     reset_button.button('Reset')

    # st.write(start_button)
    # Inverter
    inverter = Inverter(opt, domain=domain)
    img_path = img_paths[sample_img_name]
    run(inverter=inverter, img_path=img_path)








