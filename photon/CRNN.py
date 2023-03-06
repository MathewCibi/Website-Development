from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_xml(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
dataset = CustomImageDataset("./svt1/train.xml", "./svt1", transform=transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
img, label = next(iter(dataloader))
img.permute(2, 1, 0)
plt.imshow(img.numpy())
plt.title(label)
plt.show()