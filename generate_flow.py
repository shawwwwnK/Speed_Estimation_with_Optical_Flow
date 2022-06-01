from torchvision.io import read_video
import os
import torchvision
import torch
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from utils import InputPadder, increase_contrast_brightness
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Generate optical flow for data set.')
parser.add_argument("-p", "--datapath", default="small_data", type=str)
parser.add_argument("-s", "--sample", default=0, type=int)
parser.add_argument("-n", "--sample_size", default=0, type=int)

def get_flow(image1, image2, model):
    image1 = torch.permute(torch.from_numpy(image1), (2,0,1)).float()
    image2 = torch.permute(torch.from_numpy(image2), (2,0,1)).float()
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    output = model(image1, image2)

    output = output[-1]
    output = padder.unpad(output)
    output = output.squeeze()
    return output

def main():
    # parse args
    args = parser.parse_args()
    datapath = args.datapath
    sample = bool(args.sample)
    if sample:
        sample_size = args.sample_size

    # prepare model
    # model = torchvision.models.optical_flow.raft_small(pretrained=True)
    # model.eval()

    # prepare directories
    train_flow_path = os.path.join(datapath, "train_flow")
    val_flow_path = os.path.join(datapath, "val_flow")
    if train_flow_path.exists():
        for f in os.listdir(train_flow_path):
            os.remove(os.path.join(train_flow_path, f))
        for f in os.listdir(val_flow_path):
            os.remove(os.path.join(val_flow_path, f))
    else:
        os.makedirs(train_flow_path)
        os.makedirs(val_flow_path)

    
    
    

if __name__ == "__main__":
    main()