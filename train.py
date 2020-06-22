# Imports here
import torch, time
from torch import nn
from torch import optim
from PIL import Image    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

dirs = {'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir}

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([transforms.Resize(225),
                                 transforms.RandomRotation(45),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                 ]),
    'valid': transforms.Compose([transforms.Resize(225),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                                ]),
    'test': transforms.Compose([transforms.Resize(225),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                                ]),
}

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Build and train your network
model = models.vgg19(pretrained=True)
model


for parameter in model.parameters():
    parameter.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

model.classifier

model.cuda()

def validation(model, dataloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    model.to('cuda')
    model.eval()
    for images, labels in dataloaders['valid']:
        images = images.to('cuda')
        labels = labels.to('cuda')

        with torch.no_grad():
            output = model.forward(images)
            
            valid_loss = criterion(output, labels)
            ps = torch.exp(output)
            
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    valid_loss = valid_loss / len(dataloaders['valid'])
    accuracy = accuracy / len(dataloaders['valid'])   
    
    print('pass: valid    Loss: {:.4f}    Acc: {:.4f}'.format(
            valid_loss, accuracy)) 
    
    model.train()
    
    
def train_model(model, criterion, optimizer, num_epochs=25, device='cuda'):
    since = time.time()
    model.train()
    steps = 0
    print_every = 20
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        running_loss = 0
        accuracy = 0

        # Iterate over data.
        for images, labels in dataloaders['train']:
            steps +=1
            images = images.to('cuda')
            labels = labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            if steps % print_every == 0:
                model.eval()
                validation(model, dataloaders, criterion)

        epoch_loss = running_loss / dataset_sizes['train']
        accuracy = accuracy / len(dataloaders['train']) 
        print('epoch: {}    pass: train    Loss: {:.4f}    Acc: {:.4f}'.format(
            epoch+1, epoch_loss, accuracy))           

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criteria = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003)

eps = 5

model_ft = train_model(model, criteria ,optimizer, 5, 'cuda')

print(model_ft)

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({
            'class_to_idx': model.class_to_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint.pth')

print("model saved to checkpoint.pth")
