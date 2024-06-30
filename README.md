# Deep Imbalanced Regression to Estimate Vascular Age from PPG Data: A Novel Digital Biomarker for Cardiovascular Health

## Introduction
Welcome to our repository where we provide the PyTorch implementation of the **Dist loss**. This loss function, described in detail [here](https://arxiv.org/abs/2406.14953), leverages data distribution priors to effectively address deep imbalanced regression tasks, such as estimating vascular age from PPG data.

The figure below demonstrates the effectiveness of the **Dist loss** in addressing deep imbalanced regression tasks. Subfigures (a) and (b) show the results generated by the model using only L1 loss (with similar performance observed when using MSELoss), while subfigures (c) and (d) display the results produced by the model using the Dist loss during the training phase. Additionally, the following table highlights the effectiveness of this loss function in few-shot regions.

![Alt text](assets/performance.png)

![Alt text](assets/few_shot_region.png)

## Usage
To illustrate the usage of the **Dist loss**, we have included a straightforward example in the `example.ipynb` notebook, which demonstrates its application in a synthetic regression task. When utilizing this loss function, several key parameters must be carefully considered:

1. **`batch_size`**: Setting a very small batch size can degrade performance because such sizes are inadequate for reliably estimating the distribution of model outputs. A practical solution is to calculate the loss values over multiple batches or at the end of each epoch if your data or hardware does not support larger batch sizes.

2. **`step`**: This parameter defines the interval between discrete labels in the estimated distribution. For instance, in age estimation, a step value of 1 is appropriate, whereas a value like 10,000 might be suitable for predicting house prices. The step size directly impacts the granularity of your task.

3. **`min_label` and `max_label`**: These parameters define the theoretical range of possible labels. Any label values falling below `min_label` or above `max_label` will be assigned a probability of zero in the output distribution. Importantly, these are not merely the minimum and maximum values present in your dataset. For example, with a dataset ranging from 40 to 80 years in age, `min_label` and `max_label` might be set to 20 and 100 years, respectively.

4. **`drop_last`**: This parameter should be set to True in your training set to avoid mismatched tensor shapes, which can lead to errors.
