import os
import json
import streamlit as st
import random
from google.cloud import firestore
from google.oauth2 import service_account

import torch
from torchvision.transforms import transforms


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


################################# Transforms #################################
def get_transforms(args):
    transform_list = [transforms.ToTensor()]
    if args.resize:
        transform_list.append(transforms.Resize((args.img_size, args.img_size)))
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


################################# Custom Schedulers #################################


################################# DB Functions #################################
def get_img():
    # Authenticate to Firestore with the JSON account key.
    key_dict = json.loads(st.secrets['textkey'])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)

    # Create a reference to the Google post.
    doc_ref = db.collection('image').document()

    # And then upload some data to that reference.
    idx = ?
    doc_ref.set({'id': idx, 'status': True})



