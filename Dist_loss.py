import torch
import numpy as np
import torch.nn as nn
from fast_soft_sort.pytorch_ops import soft_sort

class DistLoss(nn.Module):
    def __init__(self, size_average=True, loss_fn='L1', loss_weight=1.0, regularization_strength=0.1, require_loss_values=False):
        """
        Initialize the DistLoss module.

        Parameters:
            size_average (bool): If True, average the loss; if False, sum the loss.
            loss_fn (str or nn.Module): The type of loss function to use:
                                        - 'L1': Use nn.L1Loss.
                                        - 'L2': Use nn.MSELoss.
                                        - Otherwise, should be a custom nn.Module for a specific loss.
            loss_weight (float): The weight to apply to the distribution loss.
            regularization_strength (float): Strength of regularization in soft_sort algorithm.
            require_loss_values (bool): Whether to return the individual loss values along with the total loss.
        """
        super(DistLoss, self).__init__()
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.regularization_strength = regularization_strength
        self.require_loss_values = require_loss_values

        # Determine the loss function based on the loss_fn parameter
        if loss_fn == 'L1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_fn == 'L2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            raise ValueError("Invalid loss function. Choose 'L1', 'L2', or provide a custom nn.Module.")

    def forward(self, inp, tar, theoretical_labels):
        """
        Compute the loss between the input and theoretical labels using soft_sort.

        Parameters:
            inp (torch.Tensor): The input tensor.
            tar (torch.Tensor): The target tensor.
            theoretical_labels (torch.Tensor): Theoretical labels tensor computed from kernel density estimation.

        Returns:
            torch.Tensor: The computed loss, and optionally the individual loss values.
        """
        # Perform soft sorting on the input tensor
        sorted_inp = soft_sort(inp.reshape(1, -1).cpu(), regularization_strength=self.regularization_strength)
        sorted_inp = sorted_inp.reshape(-1, 1).cuda()

        # Compute the distribution loss using the specified loss function
        dist_loss = self.loss_fn(sorted_inp, theoretical_labels)

        # Compute the plain loss
        plain_loss = self.loss_fn(inp, tar)
        
        # Compute the total loss
        total_loss = self.loss_weight * dist_loss + plain_loss
        
        # Return the average or sum of the loss based on size_average
        if self.size_average:
            if self.require_loss_values:
                return total_loss.mean(), dist_loss.mean(), plain_loss.mean()
            else:
                return total_loss.mean()
        else:
            if self.require_loss_values:
                return total_loss.sum(), dist_loss.sum(), plain_loss.sum()
            else:
                return total_loss.sum()

# For more details on the fast and differentiable sorting algorithm, visit:
# https://github.com/google-research/fast-soft-sort
