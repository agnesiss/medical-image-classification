# medical-image-classification
In this project, we used the NIH Chest X-ray dataset and applied the Model-Agnostic Meta-Learning (MAML) method to train a CNN model. Our goal was to build a model that could still perform well with only a small amount of training data.

# dataset
We chose the NIH chest x-ray dataset, a comprehensive medical image dataset from kaggle. This dataset has a collection of 112,120 frontal-view X-ray images, with corresponding disease labels and patient information.

The dataset covers 14 different disease classes, Infiltration,Effusion, Atelectasis, Nodule, Mass, Pneumothorax, Consolidation, Pleural_Thickening, Cardiomegaly, Emphysema, Edema, Fibrosis, Pneumonia, Hernia.This rich diversity allows us for sampling different kinds of diseases for training and testing.

import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "Data_Entry_2017.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "nih-chest-xrays/data",
  file_path,
)

# preprocessing
The Jupyter Notebook executed the following core data processing tasks:

Label Engineering: Transformed raw textual diagnostic labels into a multi-label one-hot encoded format suitable for machine learning models.

Image Preprocessing and Enhancement: Applied grayscale conversion, CLAHE contrast enhancement, and brightness adjustment to the raw medical images to improve quality and feature visibility.

Data Exploration and Balancing: Performed statistical analysis and visualization of the label data. Crucially, it addressed the class imbalance issue through a combination of Undersampling (reducing the majority class) and Oversampling (increasing minority classes), yielding a dataset with a more balanced distribution aligned with research requirements.

These steps established a solid foundation for utilizing the processed images and labels in subsequent model training phases.

# models
## CNN and ResNet
The ResNet-50-based structure is expected to have stronger feature representation in the multi-label medical image classification task and capture richer details of disease images compared to the base CNN model. It is also expected to have faster convergence speed, reduce training time, and improve the practical application value of the model.
## MAML
MAML is a meta-learning method that teaches a model how to quickly adapt to new tasks with only a few training examples.

First, we define a MAML class. It takes a model and sets two learning rates: one for the inner loop (task-specific update) and one for the outer loop (meta-update across tasks). We also set the number of inner loop steps.

The inner loop (inner_loop function) updates the model’s parameters for a given task. We use the functional_call function to apply changes without altering the original model. For each step, we:
Predict outputs for the support set,
Calculate the loss using BCEWithLogitsLoss (because we are doing multi-label classification),
Compute gradients with respect to the parameters,
Update the parameters in a differentiable way.

The outer loop (outer_loop function) optimizes the model across many tasks. For each task:
We run the inner loop to get adapted parameters,
Then, we evaluate the adapted model on the query set,
We sum all the query losses and average them,

Finally, we use this average loss to update the original model's parameters.

This way, after meta-training, the model learns how to adapt quickly to new tasks by just doing a few gradient updates.

# Conclusion
In this project, we have achieved the following:
Data analysis for the NIH Chest X-ray dataset
Data processing and balancing for the dataset
Compare the performance of Vanilla CNN with ResNet on training of image
Compare the effect of image agumentation and transformation on the training of medical images
Implemented MAML for small data learning
Fine-tune a resnet for small data learning and compare that with MAML

# pip pre-prequisite packages
import random</b>
import pandas as pd</b>
import seaborn as sns</b>
import matplotlib.pyplot as plt</b>
import numpy as np</b>
import Counter from collections </b>
import torch</b>
import torch.nn as nn</b>
import torch.nn.functional as F</b>
import torchvision.transforms as transforms</b>
import Dataset, DataLoader, Subset from torch.utils.data </b>
import Image from PIL </b>
import pandas as pd</b>
import os</b>
import time</b>
import torchvision.models as models</b>
import roc_auc_score,f1_score,precision_score,recall_score from sklearn.metrics </b>
import copy</b>
import functional_call from torch.nn.utils.stateless </b>
import defaultdict from collections </b>

