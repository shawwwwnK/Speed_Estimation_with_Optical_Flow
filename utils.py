import torch
import torch.distributed as dist
import torch.nn.functional as F

def compute_metrics(flow_pred, flow_gt, valid_flow_mask=None):

    epe = ((flow_pred - flow_gt) ** 2).sum(dim=1).sqrt()
    flow_norm = (flow_gt ** 2).sum(dim=1).sqrt()

    if valid_flow_mask is not None:
        epe = epe[valid_flow_mask]
        flow_norm = flow_norm[valid_flow_mask]

    relative_epe = epe / flow_norm

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
        "f1": ((epe > 3) & (relative_epe > 0.05)).float().mean().item() * 100,
    }
    return metrics, epe.numel()

def sequence_loss(flow_preds, flow_gt, valid_flow_mask, gamma=0.8, max_flow=400):
    """Loss function defined over sequence of flow predictions"""

    if gamma > 1:
        raise ValueError(f"Gamma should be < 1, got {gamma}.")

    # exlude invalid pixels and extremely large diplacements
    flow_norm = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid_flow_mask = valid_flow_mask & (flow_norm < max_flow)

    valid_flow_mask = valid_flow_mask[:, None, :, :]

    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)

    abs_diff = (flow_preds - flow_gt).abs()
    abs_diff = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0]
    weights = gamma ** torch.arange(num_predictions - 1, -1, -1).to(flow_gt.device)
    flow_loss = (abs_diff * weights).sum()

    return flow_loss

class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    # TODO: Ideally, this should be part of the eval transforms preset, instead
    # of being part of the validation code. It's not obvious what a good
    # solution would be, because we need to unpad the predicted flows according
    # to the input images' size, and in some datasets (Kitti) images can have
    # variable sizes.

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]