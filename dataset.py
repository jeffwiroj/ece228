import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np


#Follows the convention of https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class pathDataset(Dataset):
    '''
    root_dir: path to directory of where the dataset is store
    split: specify data split --> train, val, or test
    transform: function to transform image
    '''
    
    def __init__(self,root_dir = "data",split = "train", transform = None, pretrained = False):
        super(pathDataset,self).__init__()
        self.images = np.load(f"{root_dir}/{split}_images.npy")
        self.labels = np.load(f"{root_dir}/{split}_labels.npy")
        
        
        if(not pretrained):
            self.mean,self.std = [0.740545  , 0.53298219, 0.70582885],[0.1237, 0.1768, 0.1244]
        else:# if pretrained, use mean and std of imagenet statistics
            self.mean,self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            
        transform = self.transform = transform if transform else T.ToTensor() 
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = self.images[idx]
        labels = self.labels[idx]
        
        return self.transform(images),labels