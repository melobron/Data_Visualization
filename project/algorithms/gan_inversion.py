import sys
import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import cv2
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator
from lpips.loss import LPIPS
from lpips.utils import *
from utils import *


class Inverter:
    def __init__(self, args, domain):
        super(Inverter, self).__init__()

        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Parameters
        self.latent_type = args.latent_type
        self.lr = args.lr
        self.iterations = args.iterations
        self.mean = args.mean
        self.std = args.std
        self.img_size = args.img_size
        self.step = int(math.log(self.img_size, 2)) - 2
        self.style_mean_num = args.style_mean_num
        self.alpha = args.alpha
        self.style_weight = args.style_weight
        self.lpips_alpha = args.lpips_alpha
        self.mse_beta = args.mse_beta

        # Model
        self.G = StyledGenerator().to(self.device)
        # model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}(FreezeD).pth'.format(args.dataset))
        model_path = os.path.join(os.getcwd(), 'pretrained', '{}(FreezeD).pth'.format(domain))
        self.G.load_state_dict(torch.load(model_path, map_location=self.device))
        self.G.eval()

        # Mean Latent
        self.mean_style = self.get_mean_style(generator=self.G, device=self.device, style_mean_num=args.style_mean_num)

        # Transform
        self.transform = transforms.Compose(get_transforms(args))

        # Criterion
        self.lpips_criterion = LPIPS(device=self.device, net_type='alex').to(self.device).eval()
        self.MSE_criterion = nn.MSELoss().to(self.device)

    def read_img(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

    def initial_latent(self, latent_type):
        if latent_type == 'randn':
            return torch.randn((1, 512)).to(self.device)
        elif latent_type == 'zero':
            return torch.zeros((1, 512)).to(self.device)
        elif latent_type == 'mean_style':
            return self.mean_style
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_mean_style(self, generator, device, style_mean_num):
        mean_style = None

        for _ in range(style_mean_num):
            style = generator.mean_style(torch.randn(1024, 512).to(device))
            if mean_style is None:
                mean_style = style
            else:
                mean_style += style

        mean_style /= style_mean_num
        return mean_style
