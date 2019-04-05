#!/usr/bin/env python
# coding: utf-8

# Tested and ran on Python 3.6.6, Torch version 1.0.0
# Machine: Windows 10 with GTX 1060
# In[1]:


import torch
import numpy as np
import os
import sys
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# In[2]:


TRAIN_FILE = 'trainfile.txt'
VAL_FILE = 'valfile.txt'
TEST_FILE = 'testfile.txt'

IMG_PATH = 'flowers_data/jpg/'


# In[3]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
log_interval = 1000
learning_rate = 0.01
momentum = 0.01
epochs = 30


# In[4]:


class Flower102Dataset(Dataset):
    def __init__(self, train_file, val_file, test_file, img_path, train_val_test='train', transform = None):
        self.train_labels = {}
        self.val_labels = {}
        self.test_labels = {}
        self.train_val_test = train_val_test
        self.img_path = img_path
        self.transform = transform
        ctr = 0
        train_f = open(train_file, 'r')
        for line in train_f:
            img_name, label = line.split(" ")
            self.train_labels[ctr] = (img_name, int(label))
            ctr += 1
        ctr = 0
        val_f = open(val_file, 'r')
        for line in val_f:
            img_name, label = line.split(" ")
            self.val_labels[ctr] = (img_name, int(label))
            ctr += 1
        ctr = 0
        test_f = open(test_file, 'r')
        for line in test_f:
            img_name, label = line.split(" ")
            self.test_labels[ctr] = (img_name, int(label))
            ctr += 1
            
    def __len__(self):
        if self.train_val_test == 'train':
            return len(self.train_labels)
        elif self.train_val_test == 'val':
            return len(self.val_labels)
        elif self.train_val_test == 'test':
            return len(self.test_labels)
        
    def __getitem__(self, idx):
        image = None
        target = None
        if self.train_val_test == 'train':
            filename, target = self.train_labels[idx]
            image = io.imread(self.img_path + filename)
            if(len(image.shape) == 2):
                image = np.stack((image,)*3, axis=-1)
            if self.transform:
                image = self.transform(image)
                target = torch.Tensor([target]*5)
                target = target.long()
        elif self.train_val_test == 'val':
            filename, target = self.val_labels[idx]
            image = io.imread(self.img_path + filename)
            if(len(image.shape) == 2):
                image = np.stack((image,)*3, axis=-1)
            if self.transform:
                image = self.transform(image)
        elif self.train_val_test == 'test':
            filename, target = self.test_labels[idx]
            image = io.imread(self.img_path + filename)
            if(len(image.shape) == 2):
                image = np.stack((image,)*3, axis=-1)
            if self.transform:
                image = self.transform(image)
        return (image, target)
            


# In[5]:


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    total_data = len(train_loader.dataset)
    for data, target in train_loader:
        data = data.view([-1,data.shape[-3],data.shape[-2],data.shape[-1]])  
        data, target = data.to(device), target.to(device)
        target = target.view([-1])
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset),
        100. * len(train_loader.dataset) / len(train_loader.dataset), loss.item()))
    return loss


# In[25]:


def test(model, device, test_loader, log_freq = 100):
    model.eval()
    test_loss = 0
    correct = 0
    total_num_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
        
            pred = output.argmax(dim=1, keepdim=True)
        
            correct += pred.eq(target.view_as(pred)).sum().item()
                 
    accuracy = 100. * correct / total_num_data
    print('\nPerformance: Accuracy: {}/{} ({:.2f}%), Loss: {:.6f}\n'.format(
        correct, total_num_data, accuracy, test_loss))
    return test_loss, accuracy


# In[7]:


traintime_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(280),
        transforms.FiveCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])


# In[8]:


centercropnorm_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# In[9]:


flowers102train = Flower102Dataset(TRAIN_FILE, VAL_FILE, TEST_FILE, IMG_PATH, 'train', traintime_transform)
train_loader = DataLoader(flowers102train, batch_size=4,
                        shuffle=True, num_workers=0)


# In[10]:


flowers102val = Flower102Dataset(TRAIN_FILE, VAL_FILE, TEST_FILE, IMG_PATH, 'val', centercropnorm_transform)
val_loader = DataLoader(flowers102val, batch_size=4,
                        shuffle=True, num_workers=0)


# In[11]:


flowers102test = Flower102Dataset(TRAIN_FILE, VAL_FILE, TEST_FILE, IMG_PATH, 'test', centercropnorm_transform)
test_loader = DataLoader(flowers102test, batch_size=4,
                        shuffle=True, num_workers=0)


# In[ ]:


print("Model without loading weights")


# In[12]:


# No weights
resnet18 = models.resnet18(pretrained=False)
resnet18.cuda()
optimizer = optim.SGD(resnet18.parameters(), lr=learning_rate, momentum=momentum)


# In[13]:


# No weights
x_epoch = []
train_loss_ot = []
val_loss_ot = []
val_acc_ot = []
best_loss = sys.maxsize
for epoch in range(1, epochs + 1):
    # training phase
    train_loss = train(resnet18, device, train_loader, optimizer, epoch, log_interval)
    x_epoch.append(epoch)
    train_loss_ot.append(train_loss)
    
    # validation phase
    current_loss, accuracy = test(resnet18, device, val_loader, log_interval)
    val_loss_ot.append(current_loss)
    val_acc_ot.append(accuracy)
    if current_loss <= best_loss:
        torch.save(resnet18.state_dict(),"flower102noweights.pt")
        best_loss = current_loss
        
        
print("Best Loss: {}".format(best_loss))


# In[15]:


resnet18best = models.resnet18(pretrained=False)
resnet18best.cuda()
resnet18best.load_state_dict(torch.load("flower102noweights.pt", map_location=device))
loss = test(resnet18best, device, test_loader, log_interval)


# In[16]:


plt.title("Train loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, train_loss_ot)
plt.show()


# In[17]:


plt.title("Validation loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, val_loss_ot)
plt.show()


# In[24]:


plt.title("Validation Accuracy over epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x_epoch, val_acc_ot)
plt.show()


# In[ ]:


print("Model without loading weights and training all levels")


# In[28]:


# Full weights
resnet18loaded = models.resnet18(pretrained=True)
resnet18loaded.cuda()
optimizer = optim.SGD(resnet18loaded.parameters(), lr=learning_rate, momentum=momentum)


# In[29]:


# Full weights
x_epoch = []
train_loss_ot = []
val_loss_ot = []
val_acc_ot = []
best_loss = sys.maxsize
for epoch in range(1, epochs + 1):
    # training phase
    train_loss = train(resnet18loaded, device, train_loader, optimizer, epoch, log_interval)
    x_epoch.append(epoch)
    train_loss_ot.append(train_loss)
    
    # validation phase
    current_loss, accuracy = test(resnet18loaded, device, val_loader, log_interval)
    val_loss_ot.append(current_loss)
    val_acc_ot.append(accuracy)
    if current_loss <= best_loss:
        torch.save(resnet18loaded.state_dict(),"flower102fullweights.pt")
        best_loss = current_loss
        
        
print("Best Loss: {}".format(best_loss))


# In[30]:


resnet18best = models.resnet18(pretrained=False)
resnet18best.cuda()
resnet18best.load_state_dict(torch.load("flower102fullweights.pt", map_location=device))
loss = test(resnet18best, device, test_loader, log_interval)


# In[31]:


plt.title("Train loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, train_loss_ot)
plt.show()


# In[32]:


plt.title("Validation loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, val_loss_ot)
plt.show()


# In[33]:


plt.title("Validation Accuracy over epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x_epoch, val_acc_ot)
plt.show()


# In[ ]:


print("Model without loading weights and training on last 2 layers")


# In[35]:


# Full weights with freeze
resnet18freeze = models.resnet18(pretrained=True)
for p in resnet18freeze.conv1.parameters():
    p.requires_grad_(False)
for p in resnet18freeze.layer1.parameters(): #freeze layer 1
    p.requires_grad_(False)
for p in resnet18freeze.layer2.parameters(): # freeze Layer 2
    p.requires_grad_(False)
for p in resnet18freeze.layer3.parameters():
    p.requires_grad_(False)
resnet18freeze.cuda()
optimizer = optim.SGD(resnet18freeze.parameters(), lr=learning_rate, momentum=momentum)


# In[36]:


# Full weights with freeze
x_epoch = []
train_loss_ot = []
val_loss_ot = []
val_acc_ot = []
best_loss = sys.maxsize
for epoch in range(1, epochs + 1):
    # training phase
    train_loss = train(resnet18freeze, device, train_loader, optimizer, epoch, log_interval)
    x_epoch.append(epoch)
    train_loss_ot.append(train_loss)
    
    # validation phase
    current_loss, accuracy = test(resnet18freeze, device, val_loader, log_interval)
    val_loss_ot.append(current_loss)
    val_acc_ot.append(accuracy)
    if current_loss <= best_loss:
        torch.save(resnet18freeze.state_dict(),"flower102freeze.pt")
        best_loss = current_loss
        
        
print("Best Loss: {}".format(best_loss))


# In[37]:


resnet18best = models.resnet18(pretrained=False)
resnet18best.cuda()
resnet18best.load_state_dict(torch.load("flower102freeze.pt", map_location=device))
loss = test(resnet18best, device, test_loader, log_interval)


# In[38]:


plt.title("Train loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, train_loss_ot)
plt.show()


# In[39]:


plt.title("Validation loss over epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_epoch, val_loss_ot)
plt.show()


# In[40]:


plt.title("Validation Accuracy over epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x_epoch, val_acc_ot)
plt.show()


# In[ ]:





# In[ ]:




