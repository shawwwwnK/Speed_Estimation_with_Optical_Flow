import torch
import torch.nn as nn


class OpticalFlowRegression_conv(nn.Module):
    def __init__(self, size):
        super(OpticalFlowRegression_conv, self).__init__()
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

class OpticalFlowRegression_tan(nn.Module):
    def __init__(self, size):
        super(OpticalFlowRegression_tan, self).__init__()
        self.H = size[0]
        self.W = size[1]

        pool_size = 4
        self.pool1 = nn.AvgPool2d(pool_size, stride=pool_size)
        H_pool = (self.H - pool_size) // pool_size + 1
        W_pool = (self.W - pool_size) // pool_size + 1
        self.flatten = nn.Flatten()
        
        self.linear_x = nn.Linear(H_pool * W_pool, H_pool * W_pool)
        self.linear_y = nn.Linear(H_pool * W_pool, H_pool * W_pool)
        
        self.linear_tan = nn.Linear(H_pool * W_pool, H_pool * W_pool)

        self.linear_pitch = nn.Linear(H_pool * W_pool, H_pool * W_pool)
        self.linear_yaw = nn.Linear(H_pool * W_pool, H_pool * W_pool)

        self.pool2= nn.AvgPool1d(H_pool * W_pool)

    def forward(self, x):               # x shape is Batch_size, 2, H, W
        flow_x = x[:,0, :, :] 
        flow_y = x[:,1, :, :] 
        pool1_out_x = self.pool1(flow_x)
        pool1_out_y = self.pool1(flow_y)
        linear_out_x = self.linear_x(self.flatten(pool1_out_x))
        linear_out_y = self.linear_y(self.flatten(pool1_out_y))
        
        tan = linear_out_x / linear_out_y
        linear_out_tan = self.linear_tan(tan)
        
        arctan = torch.atan(linear_out_tan)

        out_pitch = self.linear_pitch(arctan)
        out_yaw = self.linear_yaw(arctan)
        
        out_pitch = self.pool2(out_pitch)
        out_yaw = self.pool2(out_yaw)
        
    
        return torch.hstack((torch.abs(out_pitch), torch.abs(out_yaw)))
