
import pandas as pd
from PIL import Image
import json
from utils import plot_image
import requests
from ast import literal_eval

import os
import matplotlib.pyplot as plt
import numpy as numpy
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
"""
#URL = requests.get("https://www.collegeboard.org/")
#jsonURL = requests.get("http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json",timeout=60)
jsonURL = requests.get("http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json", verify = False, timeout=None)
print(jsonURL.raise_for_status())
print(jsonURL)
labels = jsonURL.json()
print(labels)


header = {content_encoding = gzip}
zipURL = requests.get("http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.zip", headers  = header)
print(zipURL)
data = zipURL.text
print(data)
"""

dir_path = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 140
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = dir_path + r"\nightowls_training\\"
LABEL_DIR = dir_path + r"\labels.txt"
csv_path = dir_path + r"\image-name-id1.csv"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

train_dataset = VOCDataset(
    dir_path + r"\image-name-id1.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True)

annotations = pd.read_csv(csv_path)

for index, vals in annotations.iterrows():
    bboxes = literal_eval(annotations.iloc[index,2])
    print(bboxes)

for x, y in train_loader:
    print("X", x.shape, x)
    print("Y", y.shape, y)
"""
      x = x.to(DEVICE)
      for idx in range(BATCH_SIZE):
          if idx == BATCH_SIZE-1:
              bboxes = cellboxes_to_boxes(model(x))
              print(bboxes)
              bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
              print(bboxes)
              plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
      import sys
      sys.exit()

"""
