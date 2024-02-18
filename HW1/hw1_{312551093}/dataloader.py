import pandas as pd
from scipy import ndimage
import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms


def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        name = df['name'].tolist()
        path = ['./dataset/train/' + item for item in name]
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        name = df['name'].tolist()
        path = ['./dataset/train/' + item for item in name]
        label = df['label'].tolist()
        return path, label
    
    else:
        name = []
        for i in range(120):
            name.append('./dataset/test/'+str(i)+'.jpg')
        return name, None
        
    




class ImageLoader(data.Dataset):
    def __init__(self, mode):

        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  
        
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(3, sigma=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else :
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        # 獲取圖片路徑並加載圖片
        path = self.img_name[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        


        # 獲取真實標籤
        if self.mode == 'train' or self.mode == 'valid':
            label = self.label[index]
            return img, label
        else:
            return img
