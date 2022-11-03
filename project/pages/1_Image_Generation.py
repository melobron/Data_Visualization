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
from algorithms.image_generation import *


############################## Arguments ##############################
parser = argparse.ArgumentParser(description='Test StyleGAN')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
# parser.add_argument('--model_name', default='Dog(FreezeD).pth', type=str)
# parser.add_argument('--dataset_name', default='Dog', type=str)  # FFHQ, Dog
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


############################## Streamlit ##############################
if __name__ == '__main__':
    st.title('Image Generation')
    st.sidebar.title('Choose Variables')

    # Domain Select Box
    domain = st.sidebar.selectbox(label='Select Domain',
                                  options=['Dog', 'Cat', 'AFAD'])

    # Random Seed Slider
    random_seed = st.sidebar.slider('Random Seed', 0, 100, 50, 1)
    n_rows = st.sidebar.slider('Number of Rows', 1, 5, 3, 1)
    n_cols = st.sidebar.slider('Number of Columns', 1, 5, 3, 1)
    image = image_generation(opt, seed=random_seed, domain=domain, n_rows=n_rows, n_cols=n_cols)

    st.image(image/255., use_column_width=True)
