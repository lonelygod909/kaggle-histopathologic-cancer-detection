# create dataloader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from predict_head import PredictHead
from tqdm import tqdm
from sklearn.model_selection import KFold
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score 

class TumorDataset(Dataset):
    def __init__(self, image_dir, train_labels, transform=None):
        self.image_dir = image_dir
        self.train_labels = train_labels
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        # image is tif
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = self.image_filenames[idx]
        if filename not in self.train_labels:
            raise KeyError(f"找不到文件 '{filename}' 的标签。可用的键: {list(self.train_labels.keys())[:5]}...")
        
        label = self.train_labels[filename]
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        return image, label

class CustomCollate:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, batch):
        images, labels = zip(*batch)
        
        return torch.stack(images), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

