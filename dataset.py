import torch
import torchvision
import numpy as np
from utils import *
import os
import random

class FlowSpeedData_new(torch.utils.data.Dataset):
    def __init__(self, flow_path, label_path, down_sample_rate):
        super().__init__()
        self.flow = []
        self.label = []
        all_flow = sorted(os.listdir(flow_path))
        for flo in all_flow:
            flow_tensor = read_flow(os.path.join(flow_path, flo))
            
            # down sampling the flow
            _, image_H, image_W = flow_tensor.shape
            resizer = torchvision.transforms.Resize((image_H // down_sample_rate, image_W // down_sample_rate))
            down_flow = resizer(flow_tensor)
            self.flow.append(flow_tensor)
                   
            file_name = flo[:20]
            sample_id = int(flo[-8:-4])
            for l in os.listdir(label_path):
                if l.startswith(file_name):
                    label_arr = torch.load(os.path.join(label_path, l))
                    label_arr = label_arr["speed"]
                    speed = label_arr[sample_id]
                    self.label.append(speed)
            
    def __len__(self):
        return len(self.flow)
        
    def __getitem__(self, i):
        return self.flow[i], self.label[i]



### ### disgarded. Process data from the old small data set
def get_flow_label(flow_path, label_path, sample_rate, down_sample_rate):
    flow = []
    label = []
    all_flow = sorted(os.listdir(flow_path))
    for f in sorted(os.listdir(label_path)):
        if f.endswith(".txt"):
            f_num = f[0]
            
            # filter the label
            cur_label = np.loadtxt(os.path.join(label_path, f))
            enum = np.arange(cur_label.shape[0]).reshape(-1,1)
            cur_label = np.hstack((enum, cur_label))
            cur_label = cur_label[:-2:sample_rate]
            cur_label = np.array([row for row in cur_label if not np.isnan(row[1])])
            
            # find flows that match the label
            for l in cur_label:
                label.append(l[1:])
                valid_flow = ""
                for flow_f in all_flow:
                    if flow_f[0] == f_num and int(flow_f[2:6]) == int(l[0]):
                        valid_flow = flow_f
                flow_tensor = read_flow(os.path.join(flow_path, valid_flow))
                
                # down sampling the flow
                _, image_H, image_W = flow_tensor.shape
                resizer = torchvision.transforms.Resize((image_H // down_sample_rate, image_W // down_sample_rate))
                down_flow = resizer(flow_tensor)
                flow.append(down_flow)
                
    return flow, label

### disgarded. Process data from the old small data set
class FlowSpeedData(torch.utils.data.Dataset):
    def __init__(self, flow_path, label_path, sample_rate, down_sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.flow_path = flow_path
        self.label_path = label_path
        self.flow, self.label = get_flow_label(flow_path, label_path, sample_rate, down_sample_rate)
    
    def __len__(self):
        return len(self.flow)
        
    def __getitem__(self, i):
        return self.flow[i], self.label[i]

class TKitti(torchvision.datasets.KittiFlow):
    def __init__(self, root):
        super().__init__(root=root)
        
    def __getitem__(self, index):
        img1, img2, flow, valid_flow_mask = super().__getitem__(index)
        if (random.random() > 0.5):
            change_brightness = random.uniform(0.5, 1.5)
            img1 = torchvision.transforms.functional.adjust_brightness(img1, change_brightness)
            img2 = torchvision.transforms.functional.adjust_brightness(img2, change_brightness)
        
        if (random.random() > 0.5):
            change_contrast = random.uniform(0.5, 1.5)
            img1 = torchvision.transforms.functional.adjust_contrast(img1, change_contrast)
            img2 = torchvision.transforms.functional.adjust_contrast(img2, change_contrast)
        
        flow = torch.from_numpy(flow)
        valid_flow_mask = torch.from_numpy(valid_flow_mask)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        
        height = 368
        width = 1232
        img1 = torchvision.transforms.functional.crop(img1, 0, 0, height, width)
        img2 = torchvision.transforms.functional.crop(img2, 0, 0, height, width)
        flow = torchvision.transforms.functional.crop(flow, 0, 0, height, width)
        valid_flow_mask = torchvision.transforms.functional.crop(valid_flow_mask, 0, 0, height, width)
        return img1, img2, flow, valid_flow_mask

