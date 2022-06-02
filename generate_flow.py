from torchvision.io import read_video
import os
import torchvision
import torch
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Generate optical flow for data set.')
parser.add_argument("-p", "--datapath", default="small_data", type=str)
parser.add_argument("-r", "--sample_rate", default=2, type=int)
parser.add_argument("-s", "--sample", default=0, type=int)
parser.add_argument("-n", "--sample_size", default=0, type=int)

# brightness and contrast enhancement should be between [-127, 127]
BRIGHTNESS = 60
CONTRAST = 127

def get_flow(image1, image2, model, device):
    image1 = torch.permute(torch.from_numpy(image1), (2,0,1)).float().to(device)
    image2 = torch.permute(torch.from_numpy(image2), (2,0,1)).float().to(device)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    output = model(image1, image2)

    output = output[-1]
    output = padder.unpad(output)
    output = output.squeeze()
    return output

def save_flow_files(video_path, flow_path, model, sample_rate, device):
    for f in os.listdir(video_path):
        if f.endswith(".hevc"):
            video_file_path = os.path.join(video_path, f)
            cap = cv.VideoCapture(cv.samples.findFile(video_file_path))
            ret, image1 = cap.read()
            image2 = None
            frame_num = 0
            while ret:
                for _ in range(sample_rate - 1):
                    ret, _ = cap.read()
                    if not ret:
                        break
                if not ret:
                    break
                ret, image2 = cap.read()
                if image2 is not None:
                    image1 = increase_contrast_brightness(image1, CONTRAST, BRIGHTNESS)
                    image2 = increase_contrast_brightness(image2, CONTRAST, BRIGHTNESS)
                    flow = get_flow(image1, image2, model, device)
                    flow = flow.cpu()
                    write_flow(flow, os.path.join(flow_path, f[:-5] + '_' + str(frame_num).rjust(4, '0') + ".flo"))
                    print(f'Written frame {frame_num} from {os.path.join(video_path, f)}')
                    frame_num += sample_rate
                    image1 = image2.copy()


def main():
    # parse args
    args = parser.parse_args()
    datapath = args.datapath
    sample = bool(args.sample)
    sample_rate = args.sample_rate
    if sample:
        sample_size = args.sample_size

    # prepare model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torchvision.models.optical_flow.raft_small(pretrained=True)
    model.eval()
    model.to(device)

    # prepare directories
    train_path = os.path.join(datapath, "train")
    val_path = os.path.join(datapath, "val")
    train_flow_path = os.path.join(datapath, "train_flow")
    val_flow_path = os.path.join(datapath, "val_flow")
    if os.path.exists(train_flow_path):
        for f in os.listdir(train_flow_path):
            os.remove(os.path.join(train_flow_path, f))
        for f in os.listdir(val_flow_path):
            os.remove(os.path.join(val_flow_path, f))
    else:
        os.makedirs(train_flow_path)
        os.makedirs(val_flow_path)

    save_flow_files(train_path, train_flow_path, model, sample_rate, device)
    save_flow_files(val_path, val_flow_path, model, sample_rate, device)

                

if __name__ == "__main__":
    main()