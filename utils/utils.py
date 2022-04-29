import torch
import torch.nn as nn
import torchvision
import yaml


def get_config(path):
    with open(path,"r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


class ResNet(nn.Module):
    def __init__(self,num_class = 9,pretrained = False):
        super(ResNet,self).__init__()
        self.backbone = get_backbone(pretrained)
        self.fc = nn.Linear(2048,num_class)
    def forward(self,x):
        B = x.size(0)
        out = self.backbone(x).view(B,-1)
        return self.fc(out)



def get_backbone(pretrained = False):
    '''
        Extracts Backbone of ResNet 50
        if pretrained = True, returns backbone pretrained using VicReg on ImageNet
                              else: return Resnet with Random weight
    '''
    if(pretrained): backbone = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    else: 
        resnet = torchvision.models.resnet50(False)
        backbone = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    return backbone

def val(model,dataloader,criterion,device = "cpu"):
    
    model.eval() # Put in Eval Mode
    model = model.to(device)
    
    total,correct = 0,0
    avg_loss = 0
    with torch.no_grad():
        for x,y in dataloader:

            B = x.size(0)
            x,y = x.to(device),y.view(B).long().to(device)
            logits = model(x)
            preds = torch.argmax(logits,1)

            loss = criterion(logits,y)
            avg_loss += (loss.item() / len(dataloader))
            total += B
            correct += (preds == y).sum().item()
 
    model.train()
    return avg_loss, (correct/total)


def train_n_val(model,train_loader,val_loader,optimizer,criterion,scheduler = None,device = "cpu"):
    avg_loss,total,correct = 0,0,0
    model = model.to(device)
    for x,y in train_loader:
        
        B = x.size(0)
        x,y = x.to(device),y.view(B).long().to(device)
        logits = model(x)
        preds = torch.argmax(logits,1)
        
        loss = criterion(logits,y)
        avg_loss += (loss.item() / len(train_loader))
        total += B
        correct += (preds == y).sum().item()
        
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
    if(scheduler != None): scheduler.step()
    val_loss,val_acc = val(model,val_loader,criterion,device)
    return avg_loss, correct/total,val_loss,val_acc
        
        
        

