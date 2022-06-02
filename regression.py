import torch
import torch.nn as nn

class OpticalFlowRegression(nn.Module):
    def __init__(self, size):
        super(OpticalFlowRegression, self).__init__()
        self.H = size[0]
        self.W = size[1]
        self.scale = self.H*self.W
        self.linear_x = nn.Linear(self.scale, self.scale)
        self.linear_y = nn.Linear(self.scale, self.scale)
        self.pool = nn.AvgPool1d(self.scale)

    def forward(self, x):               # x shape is Batch_size, 2, H, W
        flow_x = x[:,0,:,:].flatten()
        flow_y = x[:,1,:,:].flatten()
        linear_out_x = self.linear_x(flow_x)
        v_pred_x = self.pool(linear_out_x) * self.scale
        linear_out_y = self.linear_y(flow_y)
        v_pred_y = self.pool(linear_out_y) * self.scale
        return torch.sqrt(v_pred_x**2 + v_pred_y**2)
