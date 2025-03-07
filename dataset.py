import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import pandas as pd

class ForenSynths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.data = []

        # Iterate over the categories
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)

            # Iterate over class names (real/fake)
            for class_name in self.classes:
                class_path = os.path.join(category_path, class_name)

                # Iterate over files
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)

                    # Append a tuple (file_path, class_index)
                    self.data.append((file_path, self.classes.index(class_name)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label

class OjhaCVPR23(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.fake_index = 1  # Index of 'fake' in ['real', 'fake']
        self.real_index = 0  # Index of 'real' in ['real', 'fake']

        sub_folders = os.listdir(root_dir)
        
        if '1_fake' in sub_folders and '0_real' in sub_folders:
            # This is the 'biggan' case
            fake_dir = os.path.join(root_dir, '1_fake')
            real_dir = os.path.join(root_dir, '0_real')
            self._process_folder(fake_dir, self.fake_index)
            self._process_folder(real_dir, self.real_index)
        else:
            # This is the 'cyclegan' case
            for folder in sub_folders:
                fake_dir = os.path.join(root_dir, folder, '1_fake')
                real_dir = os.path.join(root_dir, folder, '0_real')
                self._process_folder(fake_dir, self.fake_index)
                self._process_folder(real_dir, self.real_index)
            
    def _process_folder(self, folder_path, index):
        # Iterate over files
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Append a tuple (file_path, index)
            self.data.append((file_path, index))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label

class Wang_CVPR20(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.fake_index = 1  # Index of 'fake' in ['real', 'fake']
        self.real_index = 0  # Index of 'real' in ['real', 'fake']

        sub_folders = os.listdir(root_dir)
        
        if '1_fake' in sub_folders and '0_real' in sub_folders:
            # This is the 'biggan' case
            fake_dir = os.path.join(root_dir, '1_fake')
            real_dir = os.path.join(root_dir, '0_real')
            self._process_folder(fake_dir, self.fake_index)
            self._process_folder(real_dir, self.real_index)
        else:
            # This is the 'cyclegan' case
            for folder in sub_folders:
                fake_dir = os.path.join(root_dir, folder, '1_fake')
                real_dir = os.path.join(root_dir, folder, '0_real')
                self._process_folder(fake_dir, self.fake_index)
                self._process_folder(real_dir, self.real_index)
            
    def _process_folder(self, folder_path, index):
        # Iterate over files
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Append a tuple (file_path, index)
            self.data.append((file_path, index))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label
        

class ArtiFact(Dataset):
    def __init__(self, root_dir, label, load_percentage=100, transform=None, AL_name = False):
        self.root_dir = root_dir
        self.metadata = pd.DataFrame()
        self.label = label
        self.AL_name = AL_name
    
        if os.path.isdir(root_dir):
            subdir_metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
            subdir_metadata['image_path'] = subdir_metadata['image_path'].apply(lambda x: os.path.join(root_dir, x))
             # Selecionar uma porcentagem dos dados
            if load_percentage < 100:
                sample_size = int(len(subdir_metadata) * (load_percentage / 100))
                self.metadata = subdir_metadata.sample(sample_size, random_state=42)
            else:
                self.metadata = subdir_metadata


        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        #declare the name here
        # name = ...
    
        if self.transform:
            image = self.transform(image)
        
        if self.AL_name:
            return image, name

        return image, self.label

class AllArtiFact(Dataset):
    def __init__(self, root_dir, set, return_name=False, load_percentage=100, transform=None):
        self.root_dir = root_dir
        self.metadata = pd.DataFrame()
        self.return_name = return_name
    
        for i in os.listdir(root_dir):
          sub_dataset = os.path.join(root_dir, i)
        
          subdir_metadata = pd.read_csv(os.path.join(sub_dataset, 'metadata.csv'))
          subdir_metadata['image_path'] = subdir_metadata['image_path'].apply(lambda x: os.path.join(sub_dataset, x))
          
          self.metadata.append(subdir_metadata)
          
        
        # embaralha o dataset
        self.metadata = df.sample(frac=1).reset_index(drop=True)
        sample_size = int(len(subdir_metadata) * (0.8))
          
        if set == 'train':
            self.metadata = self.metadata.iloc[:sample_size]
        else:
            self.metadata = self.metadata.iloc[sample_size:]
        

        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['image_path']
        
        real_sets = ['afhq','celebAHQ','coco','ffhq','imagenet','landscape','lsun','metfaces','cycle_gan']
        
        is_true = any(set in img_path for set in real_sets)
        

        if is_true:
          label = 0
        else:
          label = 1
        
        image = Image.open(img_path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)
            
        if self.return_name:
            return image, img_path
         
        return image, label
        

class ArtiFactBadge(Dataset):
    def __init__(self, paths):
        self.datapaths = paths
    
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        img_path = self.datapaths[idx]
        
        real_sets = ['afhq','celebAHQ','coco','ffhq','imagenet','landscape','lsun','metfaces','cycle_gan']
        
        is_true = any(set in img_path for set in real_sets)
        
        if is_true:
          label = 0
        else:
          label = 1
        
        image = Image.open(img_path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)
            
        return image, label
                            
        