import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np

# img_dir = './assets'
# img_paths = glob(os.path.join(img_dir, '*.png'))
# for img_p in img_paths:
#     name = os.path.basename(img_p)
#     img = cv2.imread(img_p, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.resize(img, (256, 256, 3))
#     cv2.imwrite('./assets/{}'.format(name), img)
