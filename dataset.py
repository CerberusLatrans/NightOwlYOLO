"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import json
from ast import literal_eval

dir_path = os.path.dirname(os.path.realpath(__file__))

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file = dir_path + r"\image-name-id1.csv",
        img_dir = dir_path + r"\nightowls_training\\",
        label_dir = dir_path + r"\labels.txt",
        S=7, B=2, C=1, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        "this to get a list of box labels from the csv"
        boxes = literal_eval(self.annotations.iloc[index,2])
        print("BOXES LABELS FROM CSV:" + str(boxes))

        "this is to create the image input"
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        #this is for any transformations or data augmentation
        #(which is why boxes should be converted to tensor since it will probably be done in pytorch)
        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        "this is to create the label matrix in the shape of (SxSx(C+5*B)) aka (7x7x11) from the box label list"
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)  #make sure again that class label is an int

            # i,j represents the cell row and cell column
            #this takes the x,y coords (relative to whole image res) and returns the correct corresponding grid cell in the 7x7 grid
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i #make the coordinates relative to the cell by removing cells that it's not in (removing j and i from x and y coord)

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j (if Pc for box1 is 0)
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 1] == 0:
                # Set that there exists an object
                label_matrix[i, j, 1] = 1

                # Box coordinates (relative to cell)
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 2:6] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label-1] = 1
        return image, label_matrix
