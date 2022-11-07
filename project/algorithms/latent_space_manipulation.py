import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA, IncrementalPCA
import fbpca

import sys
import pickle
import os
import argparse
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator


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


def save_mean_style(args):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    generator = StyledGenerator().to(device)
    model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}(FreezeD).pth'.format(args.domain))
    # model_path = os.path.join(os.getcwd(), 'pretrained', '{}(FreezeD).pth'.format(domain))
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Mean style
    mean_style = get_mean_style(generator, device, style_mean_num=args.style_mean_num).cpu()
    data = {'mean_style': mean_style}

    # Save pickle data
    with open('../pickle_data/mean_style({}).pickle'.format(args.domain), 'wb') as f:
        pickle.dump(data, f)


def save_pca_components(args):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    generator = StyledGenerator().to(device)
    model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}(FreezeD).pth'.format(args.domain))
    # model_path = os.path.join(os.getcwd(), 'pretrained', '{}(FreezeD).pth'.format(domain))
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Style
    latent = torch.randn(args.latent_batch, 512).to(device)
    style = generator.get_style(latent)
    style = style.detach().cpu()

    # PCA
    transformer = PCA(n_components=args.n_components, svd_solver='full')
    transformer.fit(X=style)

    components = transformer.components_
    std = np.dot(components, style.T).std(axis=1)
    idx = np.argsort(std)[::-1]
    std = std[idx]
    components[:] = components[idx]
    data = {'std': std, 'components': components, 'model': transformer}

    # Save pickle data
    with open('../pickle_data/pca({}).pickle'.format(args.domain), 'wb') as f:
        pickle.dump(data, f)


def explore(args, seed, domain, control_params):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Model
    generator = StyledGenerator().to(device)
    # model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}(FreezeD).pth'.format(domain))
    model_path = os.path.join(os.getcwd(), 'pretrained', '{}(FreezeD).pth'.format(domain))
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Style
    latent = torch.randn(1, 512).to(device)
    style = generator.get_style(latent)
    style = style.detach().cpu()

    # Mean Style
    mean_style = get_mean_style(generator, device, style_mean_num=args.style_mean_num)

    # Parameters
    step = int(math.log(args.img_size, 2)) - 2

    # Components
    with open('./pickle_data/components({}).pickle'.format(domain), 'rb') as f:
        data = pickle.load(f)
    components = data['components']

    # Explore
    for i, c in enumerate(control_params):
        style += components[i, :] * c
    style = style.type(torch.FloatTensor).to(device)
    img = generator.forward_from_style(style, step=step, alpha=args.alpha, mean_style=mean_style, style_weight=args.style_weight)
    img = torch.squeeze(img, dim=0)

    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(args.mean, args.std)], std=[1/s for s in args.std])
    ])

    img = transform(img)
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0., 1.) * 255.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test StyleGAN')

    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--domain', default='FFHQ', type=str)  # FFHQ, AFAD, Cat, Dog
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

    save_pca_components(opt)
    # save_mean_style(opt)

    # # Explore
    # control_params = [0, 0, 0]
    # image = explore(opt, seed=100, domain='Dog', control_params=control_params)
    # plt.imshow(image/255.)
    # plt.show()
