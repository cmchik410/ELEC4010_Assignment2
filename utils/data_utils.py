import os, glob
import pandas as pd
import torch
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ISBI2016(Dataset):
    def __init__(self, root_dir, split, csv_file, balanced = False, label_dict = None, transform = None):
        self.root_dir = root_dir
        self.split = split
        self.label_dict = label_dict
        self.balanced = balanced
        self.df= pd.read_csv(os.path.join(root_dir, csv_file), names = ["name", "label"], header = None)
        
        benign = None
        malignant = None
        if label_dict is not None:
            benign = np.array(self.df.loc[self.df["label"] == "benign", "name"])
            malignant = np.array(self.df.loc[self.df["label"] == "malignant", "name"])
        else:
            benign = np.array(self.df.loc[self.df["label"] == 0.0, "name"])
            malignant = np.array(self.df.loc[self.df["label"] == 1.0, "name"])
            
        N1 = benign.shape[0]
        N2 = malignant.shape[0]
        
        if self.balanced:
            malignant = malignant.tolist()
            for i in range(int(N1 * 0.7) - N2):
                rand_idx = np.random.randint(0, N2)
                malignant.append(malignant[rand_idx])
                
            malignant = np.array(malignant)
        
            benign = benign[np.random.permutation(N1)][0 : N1 // 2]
        
        self.imgfile = np.concatenate([benign, malignant], axis = -1)
        
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.imgfile[idx]
        
        img = Image.open(os.path.join(self.root_dir, self.split, img_name + ".jpg")).convert("RGB")
        
        img = np.array(img)

        flip = 0.0
        contrast = 0.0
        if self.split == "training":
            flip = np.random.random() / 2
            contrast = np.random.random() / 5 
            
        trans = A.Compose([
            A.Resize(width = 244, height = 244),
            A.RandomCrop(width = 224, height = 224),
            A.HorizontalFlip(p = flip),
            A.RandomBrightnessContrast(p = contrast),
        ])
            
        img = trans(image = img)["image"]
        
        if self.transform:
            img = self.transform(img)
                
        label = self.df.loc[self.df["name"] == img_name, "label"].values 
        
        if self.label_dict is not None:
            label = self.label_dict[label[0]]

        label = np.array(label).reshape(-1)
        
        label = torch.Tensor(label)
        
        label = label.type(torch.LongTensor)

        sample = {"image" : img, "label" : label}

        return sample
    
    def __len__(self):
        return self.imgfile.shape[0]


class MRI(Dataset):
    def __init__(self, root_dir, csv_file, transform = None):
        self.root_dir = root_dir
        self.labelframe = pd.read_csv(os.path.join(root_dir, csv_file), header = 0)
        self.transform = transform
        
        self.all_files = []
        for i in self.labelframe["Patient"]:
            temp = glob.glob((os.path.join(root_dir, i) + "*/*[!_mask].tif"))

            temp = [i.split('.tif')[0] for i in temp]
            
            temp = sorted(temp, key = lambda x : int(x.split('_')[-1]))
            
            for i in temp:
                self.all_files.append(i)
        
    def __getitem__(self, idx):  
          
        img = Image.open(self.all_files[idx] + ".tif").convert("RGB")
        label = Image.open(self.all_files[idx] + "_mask.tif").convert("1")

        label = transforms.ToTensor()(label)
        
        label = label.type(torch.LongTensor)
        
        if self.transform:
            img = self.transform(img)
        
        sample = {"image" : img, "label" : label}
        
        return sample
    
    def __len__(self):
        return len(self.all_files)