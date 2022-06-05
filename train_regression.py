import torch
import torchvision
import numpy as np
from utils import *
import time
import os
from regression import OpticalFlowRegression
from dataset import FlowSpeedData


def train(train_dataloader, val_dataloader, epochs, lr, reg):
    flow_tmp = next(iter(train_dataloader))
    flow_size = flow_tmp[0].shape[2:]

    epochs = 100
    lr = 6e-7
    reg = 1e-5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = OpticalFlowRegression(flow_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        print("Epoch", str(epoch + 1) + ": ", end="")
        running_tloss = 0.0
        
        for i, (flow, label) in enumerate(train_dataloader):
            flow = flow.to(device)
            label = label.to(device).to(torch.float32)
            
            model.train(True)
            
            optimizer.zero_grad()
            output = model(flow)
            
            
            loss = loss_fn(output, label).to(torch.float32)
            
            loss.backward()
            optimizer.step()
            
            running_tloss += loss.item()
        train_loss = running_tloss / (i+1)
            
        model.train(False)

        running_vloss = 0.0
        for i, (vflow, vlabel) in enumerate(val_dataloader):
            vflow = vflow.to(device)
            vlabel = vlabel.to(device).to(torch.float32)
            voutput = model(vflow)
            vloss = loss_fn(voutput, vlabel)
            running_vloss += vloss

        val_loss = running_vloss / (i+1)
            
            
        print(f'Train Loss: {train_loss}. Validation Loss: {val_loss}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return model, train_losses, val_losses



def main():
    train = "small_data/train"
    val = "small_data/val"
    train_flow = "small_data/train_flow"
    val_flow = "small_data/val_flow"
    results_path = "results/first_results.txt"
    model_path = "results/first_model.pt"
    down_sample_rate = 8
    batch_size = 16
    epochs = 100
    lr = 6e-7
    reg = 1e-5
    
    train_set = FlowSpeedData(train_flow, train, 2, down_sample_rate)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = FlowSpeedData(val_flow, val, 2, down_sample_rate)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model, train_losses, val_losses = train(train_dataloader, val_dataloader, epochs, lr, reg)

    results = {
        "train_losses" : train_losses,
        "val_losses" : val_losses,
        "epochs" : epochs,
        "lr" : lr,
        "reg" : reg,
        "down_sample_rate" : down_sample_rate,
        "batch_size" : batch_size
    }
    torch.save(results, results_path)
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()