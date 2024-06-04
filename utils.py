import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset
import os  
import shutil

def normalization(signal:np.ndarray, axis:int=1, eps:float=1e-8):
    """
    Normalize the signal.

    Parameters:
        signal: The signal to be normalized.
        axis: The axis along which to perform the normalization. Default is 1.
        eps: A small value added to the denominator to avoid division by zero. Default is 1e-8.
    """
    mean_signal = signal.mean(axis=axis, keepdims=True)
    std_signal = signal.mean(axis=axis, keepdims=True)
    normalized_signal = (signal - mean_signal) / (std_signal + eps)
    return normalized_signal

def get_ld(labels:np.ndarray, bw_method=0.5, min_label=21, max_label=100):
    """
    Get the label distribution using kernel density estimation.

    Parameters:
        labels: The label array to be operated.
        bw_method: The bandwidth of the kernel distribution estimator. 
        min_label: The minimum of the label. Values below this will be assigned a zero value.
        max_label: The maximum of the label. Values above this will be assigned a zero value. 
    """
    kde = gaussian_kde(labels, bw_method=bw_method)
    x = np.arange(min_label, max_label+1)
    density_estimation = kde(x)
    return density_estimation

def get_batch_label_distribution(density:np.ndarray, batch_size:int, mode='cumsum', range_res=30) -> np.ndarray:
    """
    Acquire the label distribution of one batch according to the kernel density estimation.

    Parameters:
        density (np.ndarray): The estimated density according to the kernel density estimation.
            This 1D array represents the density values for each label.
        batch_size (int): The number of samples in a batch.

    Returns:
        np.ndarray: An array representing the label distribution for one batch, ensuring it sums up to 'batch_size'.

    Explanation:
        This function calculates the label distribution for a batch based on the estimated density values.
        It first determines the number of samples for each label by scaling the density values with 'batch_size'.
        Then, it distributes these samples across the labels ensuring that the total sum matches 'batch_size'.
        If there's a residual difference due to rounding, it adjusts the distribution to match 'batch_size'.
    """
    num_density = density * batch_size

    if mode == 'cumsum':
        batch_label_distribution = np.zeros_like(num_density)
    else:
        batch_label_distribution = np.ones_like(num_density)

    forward_cumsum = num_density.cumsum(axis=0)
    backward_cumsum = num_density[-1::-1].cumsum(axis=0)
    forward_index = np.where(forward_cumsum >= 1)[0][0]
    backward_index = -1 * np.where(backward_cumsum >= 1)[0][0] - 1
    forward_index_cumsum = forward_cumsum[forward_index]
    backward_index_cumsum = backward_cumsum[-1*(backward_index+1)]

    batch_label_distribution[forward_index] = forward_index_cumsum.round()
    batch_label_distribution[forward_index+1:backward_index] = num_density[forward_index+1:backward_index].round()
    batch_label_distribution[backward_index] = backward_index_cumsum.round()

    sum_batch_label_distribution = batch_label_distribution.sum()
    res_sum = batch_size - int(sum_batch_label_distribution)
    maximum_index = batch_label_distribution.argmax()

    if abs(res_sum) <= range_res:
        left_index = maximum_index - abs(res_sum) // 2
        right_index = left_index + abs(res_sum)
        batch_label_distribution[left_index:right_index] += np.sign(res_sum)
    else:
        iters = res_sum // range_res
        remainder = res_sum % range_res
        left_index = maximum_index - range_res // 2
        right_index = left_index + range_res
        batch_label_distribution[left_index:right_index] += iters * np.sign(res_sum)
        batch_label_distribution[maximum_index] += remainder
    batch_label_distribution.sum()
    return batch_label_distribution

def get_batch_distribution_labels(density:np.ndarray, batch_size:int, min_label:int) -> np.ndarray:
    """"
    Acquire the labels following the label distribution of one batch.

    Parameters: 
        density (np.ndarray): The estimated density according to the kernel density estimation.
            This 1D array represents the density values for each label.
        batch_size (int): The number of samples in a batch.
        min_label (int): The minimum of the labels.
    """
    batch_distribution_label = np.zeros((batch_size))
    batch_label_distribution = get_batch_label_distribution(density, batch_size)
    cumsum_batch_label_distribution = batch_label_distribution.cumsum(axis=0).astype(int)

    for i, num in enumerate(batch_label_distribution):
        if i == 0:
            before_cumsum = 0
        else:
            before_cumsum = cumsum_batch_label_distribution[i-1]
        batch_distribution_label[before_cumsum:before_cumsum+int(num)] = i + min_label
    
    return batch_distribution_label