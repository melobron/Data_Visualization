import torch

import os
import sys
import pickle
import random
import numpy as np

models_path = os.path.dirname(os.getcwd())
sys.path.append(models_path)
from models.StyleGAN2 import StyledGenerator


def get_coord(style_vector, transformer, n_axis):
    transformed_vector = transformer.transform(style_vector)
    transformed_vector = transformed_vector[:, :n_axis]
    return transformed_vector


if __name__ == '__main__':
    domain = 'FFHQ'

    # Device
    device = torch.device('cuda:{}'.format(0))

    # Random Seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Model
    generator = StyledGenerator().to(device)
    model_path = os.path.join(os.path.dirname(os.getcwd()), './pretrained', '{}(FreezeD).pth'.format(domain))
    # model_path = os.path.join(os.getcwd(), 'pretrained', '{}(FreezeD).pth'.format(domain))
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Style
    latent = torch.randn(1, 512).to(device)
    style = generator.get_style(latent)
    style = style.detach().cpu()

    # Pickle data
    with open('../pickle_data/pca({}).pickle'.format(domain), 'rb') as f:
        pickle_data = pickle.load(f)

    # Transformer
    transformer = pickle_data['model']

    get_coord(style, transformer, n_axis=3)
