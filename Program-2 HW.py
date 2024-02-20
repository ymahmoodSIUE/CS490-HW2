#!/usr/bin/env python
# coding: utf-8

# In[321]:


import torch


# In[322]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type=='cuda':
    print(torch.cuda.get_device_name(0))


# In[323]:


CLASSES=10
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[324]:


import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np


# In[325]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4, padding=0)
        self.local_response1=nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=1,padding=2)
        self.local_response2=nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=96,out_channels=96,kernel_size=1,padding=1)
        self.local_response3=nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv4=nn.Conv2d(in_channels=96,out_channels=256,kernel_size=11,stride=4, padding=2)
        self.conv5=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1, padding=1)
        self.conv7=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=1,stride=1, padding=1)
        self.conv8=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=1,stride=1, padding=1)
        self.pool8=nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc9 = nn.Linear(in_features=(6*6*384),out_features=4096)
        self.fc10 = nn.Linear(in_features=4096,out_features=4096)
        self.fc11 = nn.Linear(in_features=4096,out_features=CLASSES)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.local_response1(self.conv1(x))))
        x = self.pool2(F.relu(self.local_response2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = F.dropout(x, 0.5)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.dropout(x, 0.5)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc9(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc10(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc11(x))
        return x


# In[326]:


transform_conf=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])


# In[327]:


BATCH_SIZE=64
train_dataset=datasets.MNIST('/users/ymahmoo/data',train=True,download=True,transform=transform_conf,)
test_dataset=datasets.MNIST('/users/ymahmoo/data',train=False,download=True,transform=transform_conf)


# In[328]:


train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)


# In[329]:


model=Net().to(device)
optimizer=optim.Adam(params=model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


# In[330]:


def train(model,device,train_loader,optimizer,epochs):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_loader):
        classes=classes.type(torch.LongTensor)
        img,classes=img.to(device),classes.to(device)
        torch.autograd.set_detect_anomaly(True)     
        optimizer.zero_grad()
        output=model(img)
        loss = loss_fn(output,classes)                
        
        loss.backward()
        optimizer.step()
    if(batch_ids +1) % 2 == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_ids* len(img), len(train_loader.dataset),
            100.*batch_ids / len(train_loader),loss.item()))


# In[331]:


def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for img,classes in test_loader:
            img,classes=img.to(device), classes.to(device)
            y_hat=model(img)
            test_loss+=F.nll_loss(y_hat,classes,reduction='sum').item()
            _,y_pred=torch.max(y_hat,1)
            correct+=(y_pred==classes).sum().item()
        test_loss/=len(test_dataset)
        print("\n Test set: Avarage loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n".format(
            test_loss,correct,len(test_dataset),100.*correct/len(test_dataset)))
        print('='*30)


# In[ ]:


if __name__=='__main__':
    seed=42
    EPOCHS=1
    
    for epoch in range(1,EPOCHS+1):
        train(model,device,train_loader,optimizer,epoch)
        test(model,device,test_loader)


# In[ ]:





# In[ ]:





# In[ ]:




