# Deep Imbalanced Regression to Estimate Vascular Age from PPG Data: a Novel Digital Biomarker for Cardiovascular Health
## Introduction
In this repo, we provide our PyTorch implementation of the **Dist loss** <https://arxiv.org/abs/2406.14953>, which is a simple yet effective loss function based on data distribution priors, designed for deep imbalanced regression tasks (e.g., vascular age estimation from PPG data).  
## Usage
Here, we provide a simple example to demonstrate the usage of the **Dist loss** in a synthetic regression task in the `example.ipynb`. There are several important parameters need to be fully considered when use this loss function.  
`(1) batch_size`: If the batch_size is set too small, the performance will be not good, as a small batch_size is not reliable to estimate the distribution of model outputs. One solution is to calculate the loss values a few batches or one epoch one time.  
`(2) step`: Step is the interval between discrete labels in the estimated distribution. For age estimation, a suitable step is 1, and maybe 10k for house price prediction. It depends on and determines the granularity of your tasks.  
`(3) min_label and max_label`: `min_label` and `max_label` represent the theoretical range of possible labels. Any label values below `min_label` or above `max_label` will have an assigned probability of zero in the output distribution. It is important to note that they are not the minimum and maxmum of your datasets. For example, if your dataset with an age ranged from 40 to 80 years, the `min_label` and `max_label` can be set to 20 to 100 years.

