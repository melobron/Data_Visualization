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
import streamlit as st

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator
from algorithms.latent_space_manipulation import *


############################## Arguments ##############################
parser = argparse.ArgumentParser(description='Test StyleGAN')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256

# Mean Style
parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

# PCA
parser.add_argument('--latent_batch', default=100000, type=int)  # Number of styles to compute PCA
parser.add_argument('--n_components', default=100, type=int)  # Number of eigenvectors

# Transformations
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()


############################## Streamlit ##############################
if __name__ == '__main__':
    st.title('Latent Space Manipulation')
    st.sidebar.title('Choose Variables')

    # Domain Select Box
    domain = st.sidebar.selectbox(label='Select Domain',
                                  options=['Dog', 'Cat', 'AFAD'])

    # Random Seed Slider
    random_seed = st.sidebar.slider('Random Seed', 0, 100, 50, 1)

    control_params = []
    for i in range(5):
        control_params.append(st.sidebar.slider('Control {}th Axis'.format(i+1), -50, 50, 0, 1) / 25.)

    image = explore(opt, seed=random_seed, domain=domain, control_params=control_params)
    st.image(image/255., use_column_width=True)
