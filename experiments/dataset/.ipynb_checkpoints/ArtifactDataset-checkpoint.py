import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class ArtiFact(Dataset):
    def __init__(self, root_dir, label, load_percentage=100, transform=None, return_name=True, data_list=None):
        self.root_dir = root_dir
    
        self.label = label

        print(return_name)
        self.return_name = return_name
        print(self.return_name)

        if data_list is None:
            self.metadata = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            self.metadata = [os.path.join(root_dir, f) for f in data_list]

        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata[idx]
        image = Image.open(img_path).convert("RGB")
        #declare the name here
        name = img_path[len(self.root_dir)+1:]
    
        if self.transform:
            image = self.transform(image)
        
        if self.return_name:
            #print("erro")
            return image, name

        return image, torch.tensor([self.label], dtype=torch.float32)