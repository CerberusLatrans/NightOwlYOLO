"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import json

dir_path = os.path.dirname(os.path.realpath(__file__))

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file = dir_path + r"\image-name-id.csv",
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
        "this to create a list of box labels"

"""
        #gets the image id of the corresponding index in the csv
        image_id = self.annotations.iloc[index, 1]
        boxes = []

        with open(self.label_dir) as f:
            #reading the json (.txt) file with the "images" and "annotations" sections
            data = json.loads(f.read())

            #finding an annotation id(s) which matches the id of the index in the csv
            #if there is more than one annotation id found, then there are multiple bounding boxes
            for label in data["annotations"]:
                if label["image_id"] == image_id:
                    #once found, it stores the class number (1,2,3,4) in class_label
                    class_label = label["category_id"]

                    #if the label is 1 (pedestrian) then it appends the the boxes list:
                    #a list of the label (can only be "1") and the box dimensions scaled by the image dimensions (1024x640)
                    if class_label == 1:
                        x,y,width,height = label["bbox"]
                        boxes.append([class_label, x/1024, y/640, width/1024, height/640])
                    elif class_label != 1:
                        pass
                    #print("BOXES", boxes)
                else:
                    pass
"""
        "this is to create the image input"
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        #this is for any transformations or data augmentation
        #(which is why boxes should be converted to tensor since it will probably be done in pytorch)
        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        "this is to create the label matrix in the shape of (SxSx(C+5*B)) from the box label list"
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)  #make sure again that class label is an int

            # i,j represents the cell row and cell column
            #this turns the coordinates x,y from relative to the whole image to relative to the grid cell
            i, j = int(self.S * y), int(self.S * x) #scale the x,y coordinates by multiplying by S
            x_cell, y_cell = self.S * x - j, self.S * y - i #make the coordinates relative to the cell by removing cells that it's not in

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
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
