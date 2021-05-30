"""
Main file for training Yolo model on Pascal VOC dataset
"""

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
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = dir_path + r"\nightowls_training\\"
LABEL_DIR = dir_path + r"\labels.txt"

J = []
n_samples = 0

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    J_train_avg = sum(mean_loss)/len(mean_loss)
    J.append(J_train_avg)
    print(f"Mean loss was {J_train_avg}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        dir_path + r"\image-name-id1.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    n_samples = train_dataset.__len__()
    print(n_samples)
    """
    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    """
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    """
    for epoch in range(EPOCHS):
        print("EPOCH: ", epoch+1)
        """
        for x, y in train_loader:
           x = x.to(DEVICE)
           for idx in range(BATCH_SIZE):
               if idx == BATCH_SIZE-1:
                   bboxes = cellboxes_to_boxes(model(x))
                   bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                   plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
           #import sys
           #sys.exit()
           """
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        print("DEBUGG PRED BOXES", len(pred_boxes), pred_boxes)
        print("DEBUGG TARGET BOXES", len(target_boxes), target_boxes)

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)

    #PLOTTING LOSS
    specifications = r"\YOLO Progress Pictures\\" + f"{J[-1]}, lr={LEARNING_RATE}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, weight_decay={WEIGHT_DECAY}, n_samples={n_samples}"
    loss_graph_path = dir_path + specifications + r'.png'
    #print(loss_graph_path)


    plt.plot(J)
    plt.title(f"{J[-1]}, lr={LEARNING_RATE}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, weight_decay={WEIGHT_DECAY}, n_samples={n_samples}")
    plt.ylabel('AVG LOSS')
    plt.xlabel('# BATCH ITERATIONS (#BATCHESperEPOCH * #EPOCHS)')
    plt.yticks(numpy.arange(0, 1, step=0.1))
    plt.savefig(loss_graph_path)
    plt.show()


if __name__ == "__main__":
    main()


#bug: non max suppression seems to be eliminating every prediction box
#this is causing the get_bboxes() function to output no predictions
#fixed by commenting out a line eliminating all boxes below the threshold value in non max suppression

#bug: the average precision list length in mAP being 0 is causing division by 0
#fixed by changing c+1 back to c so that predictions of class 0 are appended in MAP

#epoch 30 mean loss to 0.74
#bug: mAP still at 0

#of the 784 predictions, only 368, 384, 400 (increasing as it trains) have class=1 (others have class=0)
#epoch 478: 0.00116 mean loss, 0.6 mAP

#epoch 50: 0.09 Loss, 0.0 mAP (took 50 min)
#about 1 min per epoch
