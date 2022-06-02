import torch
import torch.nn as nn

class OpticalFlowRegression(nn.Module):
    def __init__(self, size):
        super(OpticalFlowRegression, self).__init__()
        self.H = size[0]
        self.W = size[1]
        self.scale = self.H*self.W
        self.pool1 = nn.AvgPool2d(8, stride=8)
        self.linear_x = nn.Linear(self.scale/8, self.scale/8)
        self.linear_y = nn.Linear(self.scale/8, self.scale/8)
        self.pool2= nn.AvgPool1d(self.scale/8)

    def forward(self, x):               # x shape is Batch_size, 2, H, W
        flow_x = x[:,0,:,:]
        flow_y = x[:,1,:,:]

        pool_out_x = self.poo1(flow_x)
        pool_out_y = self.poo1(flow_y)

        linear_out_x = self.linear_x(pool_out_x).flatten()
        linear_out_y = self.linear_y(pool_out_y).flatten()

        v_pred_x = self.pool(linear_out_x)
        v_pred_y = self.pool(linear_out_y)
        return torch.sqrt(v_pred_x**2 + v_pred_y**2)
