import torch
import torchvision
import numpy as np
from utils import *
import os

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