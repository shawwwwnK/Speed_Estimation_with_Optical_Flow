import torch
import torchvision
import numpy as np
from utils import *
import time
import os
import random
import torch.nn as nn
from dataset import TKitti

def main():
    kitti = TKitti("../cs231n/Kitti")
    finetune_epochs = 20
    finetune_loader = torch.utils.data.DataLoader(kitti, batch_size=4, shuffle=True)

    lr = 2e-5
    weight_decay = 5e-5
    eps = 1e-8
    num_train_flow_updates = 12

    raft_model = torchvision.models.optical_flow.raft_small(pretrained=True)
    optimizer = torch.optim.AdamW(raft_model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    total_steps = finetune_epochs * len(finetune_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raft_model.train()
    raft_model.to(device)

    epe_hist = []
    f1_hist = []

    for epoch in range(finetune_epochs):
        running_epe = 0.0
        running_f1 = 0.0
        for i, (image1, image2, flow_gt) in enumerate(finetune_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            flow_gt = flow_gt.to(device)
            
            optimizer.zero_grad()

            flow_predictions = raft_model(image1, image2, num_flow_updates=num_train_flow_updates)
            
            mask_shape = [flow_gt.shape[0], flow_gt.shape[2], flow_gt.shape[3]]
            valid_flow_mask = torch.ones(mask_shape).type(torch.bool).to(device)
            
            loss = sequence_loss(flow_predictions, flow_gt, valid_flow_mask)
            
            metrics, _ = compute_metrics(flow_predictions[-1], flow_gt)
            running_epe += metrics["epe"]
            running_f1 = metrics["f1"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raft_model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            
        running_epe /= (i+1)
        running_f1 /= (i+1)
        epe_hist.append(running_epe)
        f1_hist.append(running_f1)
        print(f'Epoch {epoch + 1}: EPE {running_epe}')

    raft_results = {
        "epe_hist" : epe_hist,
        "f1_hist" : f1_hist,
    }
    raft_results_path = "finetuned_raft.txt"
    torch.save(raft_results, raft_results_path)
    raft_model_path = "fintuned_raft.pt"
    torch.save(raft_model.state_dict(), raft_model_path)

if __name__ == "__main__":
    main()

