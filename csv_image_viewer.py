import torch
import os
import pandas as pd
from PIL import Image
import json
from utils import plot_image
import requests

#URL = requests.get("https://www.collegeboard.org/")
URL = requests.get("http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json")
print(URL)
data = URL.text
print(data)

dir_path = os.path.dirname(os.path.realpath(__file__))

img_dir = dir_path + r"\nightowls_training\\"
csv_path = dir_path + r"\image-name-id1.csv"
label_dir = dir_path + r"\labels.txt"

annotations = pd.read_csv(csv_path)

for index, vals in annotations.iterrows():
    image_name, image_id = vals
    print(image_name, image_id)

    img_path = os.path.join(img_dir, image_name)
    print(img_path)
    image = Image.open(img_path)
    image.show()

    boxes = []
    with open(label_dir) as f:
        data = json.loads(f.read())

        for label in data["annotations"]:
            if label["image_id"] == image_id:
                class_label = label["category_id"]

                if class_label == 1:
                    x,y,width,height = label["bbox"]
                    boxes.append([class_label, x/1024, y/640, width/1024, height/640])
                elif class_label != 1:
                    pass
            else:
                pass
    print(boxes)

    plot_image(image, boxes)
