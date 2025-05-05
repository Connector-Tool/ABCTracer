import torch
import torch.nn.functional as F


def list_wise_cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    probabilities = F.softmax(y_pred, dim=1)
    loss = -torch.sum(y_true * torch.log(probabilities + 1e-10), dim=-1)
    return loss.mean()
