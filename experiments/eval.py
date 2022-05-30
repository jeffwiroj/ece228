import sys
import yaml
import argparse
sys.path.append('../')
from utils.utils import get_config,ResNet,train_n_val,val,train_val_mx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from adamp import AdamP
from dataset import pathDataset
from attrdict import AttrDict
import os
import json
import random
import numpy as np
import torchmetrics


def parse_cfg():
    parser = argparse.ArgumentParser(description='ResNet Training Config')
    parser.add_argument('--name',default = "basic_0",type = str,help = 'yaml file name')
    args = parser.parse_args()
    
    config = get_config(f"../experiments/configs/{args.name}.yaml")
    return config






def get_corruption(corruption = "bright",scale = None):
    
    
    corrupt = None
    if(corruption == "bright"):
         corrupt = T.ColorJitter(brightness = (scale,scale))
    if(corruption == "blur"):
        corrupt = T.GaussianBlur(kernel_size = 5,sigma=(scale))
     
    
    return corrupt


def evaluate(config,model,criterion,device,corruption = "bright"):
    
    
    d = pathDataset(root_dir = "../data",pretrained = config.pretrained)
    
    
    accuracies = []
    if(corruption == "bright"):
        scales = [0.8,0.9,1.0,1.1,1.2]
    if(corruption == "blur"):
        scales = [0.3,0.5,0.7,0.9]
        
        
    stats = {corruption: []}
    f =  open(f"results/eval/{config.exp_name}.txt",'a', buffering = 1)
    
    
    for scale in scales:
        corrupt = get_corruption(corruption,scale)
        val_transform = T.Compose([
                         T.ToPILImage(),
                         corrupt,
                         T.ToTensor(),
                         T.Normalize(mean=d.mean,std=d.std)]) 
        test_dataset = pathDataset(root_dir = "../data",split = "test",transform = val_transform)
        test_loader = DataLoader(test_dataset,batch_size = config.batch_size,shuffle = True, pin_memory = True)
        _, test_acc,rms = val(model,test_loader,criterion,device = device,compute_rmse = True)
        
        accuracies.append(test_acc)
        stats[corruption].append((scale,test_acc,rms))
        
    print(json.dumps(stats),file = f)
        
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    return mean,accuracies
        
def main():
    
    torch.manual_seed(1598)
    random.seed(1598)
    np.random.seed(1598)
    
    if not os.path.exists("results"):
        print("Making results folder")
        os.makedirs("results")
    if not os.path.exists("results/weights"):
        print("Making weights folder")
        os.makedirs("results/weights")
    if not os.path.exists("results/train"):
        print("Making train result folder")
        os.makedirs("results/train")
    if not os.path.exists("results/eval"):
        print("Making train result folder")
        os.makedirs("results/eval")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = AttrDict(parse_cfg())
    model = ResNet(pretrained = False)
    model = model.to(device)
    #load model to evaluate on test set
    checkpoint = torch.load(f"results/weights/{config.exp_name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Testing on Model: {config.exp_name}")
    
    corruptions = ["bright","blur"]
    
    for corruption in corruptions:
        criterion = nn.CrossEntropyLoss()
        acc,accs = evaluate(config,model,criterion,device,corruption)
        #print(f"Corruption: {corruption}, Mean Acc: {acc}")
if __name__ == "__main__":
    main()

      
    