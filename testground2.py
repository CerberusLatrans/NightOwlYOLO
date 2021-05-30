import csv
import json
import pandas as pd
import os
from ast import literal_eval
from dataset import VOCDataset

dir_path =  os.path.dirname(os.path.realpath(__file__))

transform = None
img_dir = dir_path + r"\nightowls_training\\"
label_dir = dir_path + r"\labels.txt"
train_dataset = VOCDataset(
    dir_path + r"\image-name-id1.csv",
    transform=transform,
    img_dir=img_dir,
    label_dir=label_dir,
)

print(train_dataset[0])
