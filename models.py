import torch
import torch.nn as nn


class OpticalFlowRegression(nn.Module):
    def __init__(self, size):
        super(OpticalFlowRegression, self).__init__()
        self.H = size[0]
        self.W = size[1]
        self.conv_pitch = nn.Conv2d(2, 1, 1)
        self.conv_yaw = nn.Conv2d(2, 1, 1)
        pool_size = 4
        self.pool1 = nn.AvgPool2d(pool_size, stride=pool_size)
        H_pool = (self.H - pool_size) // pool_size + 1
        W_pool = (self.W - pool_size) // pool_size + 1
        self.flatten = nn.Flatten()
        self.linear_pitch = nn.Linear(H_pool * W_pool, H_pool * W_pool)
        self.linear_yaw = nn.Linear(H_pool * W_pool, H_pool * W_pool)
        self.pool2= nn.AvgPool1d(H_pool * W_pool)

    def forward(self, x):               # x shape is Batch_size, 2, H, W

        conv_out_pitch = self.conv_pitch(x)
        pool1_out_pitch = self.pool1(conv_out_pitch)
        flatten_out_pitch = self.flatten(pool1_out_pitch)
        linear_out_pitch = self.linear_pitch(flatten_out_pitch)
        out_pitch = self.pool2(linear_out_pitch)

        conv_out_yaw = self.conv_yaw(x)
        pool1_out_yaw = self.pool1(conv_out_yaw)
        flatten_out_yaw = self.flatten(pool1_out_yaw)
        linear_out_yaw = self.linear_yaw(flatten_out_yaw)
        out_yaw = self.pool2(linear_out_yaw)
        
    
        return torch.hstack((torch.abs(out_pitch), torch.abs(out_yaw)))
