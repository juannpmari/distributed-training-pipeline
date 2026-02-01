import torch
import torch.nn as nn


def build_loss(config):
    loss_name = config.training.loss

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
