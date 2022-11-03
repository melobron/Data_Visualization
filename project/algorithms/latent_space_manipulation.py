import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA, IncrementalPCA
import fbpca

import os
import argparse
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt








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

    tile = style_mixing(args=opt, seed=100, domain='Dog', n_source=5, n_target=3, mixing_range=(0, 3))
    plt.imshow(tile/255.)
    plt.show()