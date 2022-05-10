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




def get_model(config):
    return ResNet(pretrained = config.pretrained)
    

def freeze_bn(model):
    '''
    Freezes BatchNorm Parameters:
    '''
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d): 
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        

def parse_cfg():
    parser = argparse.ArgumentParser(description='ResNet Training Config')
    parser.add_argument('--name',default = "basic_0",type = str,help = 'yaml file name')
    args = parser.parse_args()
    
    config = get_config(f"../experiments/configs/{args.name}.yaml")
    return config


def get_dataset(config):
    d = pathDataset(root_dir = "../data",pretrained = config.pretrained)
    #print(f"Dataset mean:{d.mean}, std:{d.std}")
    if(config.transf == None):
        train_transform =  T.Compose([
                     T.ToPILImage(),
                     T.ToTensor(),
                     T.Normalize(mean=d.mean,std=d.std)])
    elif(config.transf.lower() == "basic"):
        train_transform =  T.Compose([
                     T.ToPILImage(),
                     T.RandomHorizontalFlip(0.4),
                     T.RandomVerticalFlip(0.4),
                     T.RandomApply([T.RandomRotation(20)], p=0.4),
                     T.ToTensor(),
                     T.Normalize(mean=d.mean,std=d.std)])
    val_transform = T.Compose([
                     T.ToPILImage(),
                     T.ToTensor(),
                     T.Normalize(mean=d.mean,std=d.std)])  
    
    train_dataset = pathDataset(root_dir = "../data",split = "train",transform = train_transform)
    val_dataset = pathDataset(root_dir = "../data",split = "val",transform = val_transform)
    test_dataset = pathDataset(root_dir = "../data",split = "test",transform = val_transform)
    
    return train_dataset,val_dataset,test_dataset


def save(model,optimizer,config):
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"results/weights/{config.exp_name}.pth")


        
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = AttrDict(parse_cfg())
    print("Training Configs: " , config)
    
    model = get_model(config)
    model = model.to(device)
    
    if config.pretrained == True:
        #make sure scale > 0
        if(config.lr_scale <=0): 
            config.lr_scale = 1
            
        if config.freeze_bn == True:
            freeze_bn(model)
            
        params = [{'params': model.backbone.parameters(),'lr': (config.optim.lr/config.lr_scale)},
                {'params': model.fc.parameters(), 'lr': config.optim.lr}]
        
    else: params = model.parameters()
    if(config.optim.name == "adam"):
        optimizer = torch.optim.AdamW(params,weight_decay = config.optim.wd)
        
    elif(config.optim.name == "adamp"):
         optimizer = AdamP(params, weight_decay= config.optim.wd)
            
    elif(config.optim.name == "sgd"):
        optimizer = torch.optim.SGD(params,weight_decay = config.optim.wd,momentum = 0.9)
        
    if(config.sched == "exp"):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.97)
        print("Using Scheduler")
        
    criterion = nn.CrossEntropyLoss()
    datasets = get_dataset(config)
    
    train_loader = DataLoader(datasets[0],batch_size = config.batch_size,shuffle = True, pin_memory = True)
    val_loader = DataLoader(datasets[1],batch_size = config.batch_size,shuffle = False, pin_memory = True)
    test_loader = DataLoader(datasets[2],batch_size = config.batch_size,shuffle = False, pin_memory = True)
    
    
    best_acc = -float("inf")
    f =  open(f"results/train/{config.exp_name}.txt",'a', buffering = 1)
    for epoch in range (config.epochs):
        if(config.mixup == 0):
            t_loss,t_acc,v_loss,v_acc  = \
            train_n_val(model,train_loader,val_loader,optimizer,criterion,scheduler = None,device = device)
        else: t_loss,t_acc,v_loss,v_acc  = \
              train_val_mx(model,train_loader,val_loader,optimizer,criterion,scheduler = None,device = device, mixup = config.mixup)
              
        stats = {"train_loss":t_loss, "train_acc": t_acc, "val_loss":v_loss, "val_acc":v_acc}
        
        print(f"Epoch: {epoch}, train_loss:{t_loss}, train_acc: {t_acc}, val_loss: {v_loss}, val_acc: {v_acc}")
        print(json.dumps(stats),file = f)
        
        if(v_acc > best_acc):
            best_acc = v_acc
            save(model,optimizer,config)
        
    
    
    #load best model to evaluate on test set
    checkpoint = torch.load(f"results/weights/{config.exp_name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss,test_acc = val(model,test_loader,criterion,device = device)
    print(f"Final Test Acc: {test_acc}, test_loss: {test_loss}")
    print(json.dumps({"test_acc": test_acc, "test_loss": test_loss}),file = f)
if __name__ == "__main__":
    main()

      
    