# Imports here
import torch, time, json
from torch import nn
from torch import optim
from PIL import Image    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

data_dir = input("Enter the directory where train,valid,test datasets are there ")

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



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Build and train your network
modelAvailable = ['vgg19', 'vagg16']

modelS = input("select your training model.... default will be vgg11... 0)vgg19 or 1)vgg16 ")
if modelS == 0:
	model = models.vgg19(pretrained=True)
else if modelS == 1: 
	model = models.vgg16(pretrained=True)
else:
    model = models.vgg11(pretrained=True)


for parameter in model.parameters():
    parameter.requires_grad = False

hiddenU = input("Enter no. of hidden unit ")
hiddenU = int(hiddenU)
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hiddenU)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, hiddenU)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

model.classifier

def validation(model, dataloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    model.to(device)
    model.eval()
    for images, labels in dataloaders['valid']:
        images = images.to(device)
        labels = labels.to(device)

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
            images = images.to(device)
            labels = labels.to(device)

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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

d = input("Select device for training default will be GPU...0)Cpu 1)Gpu")
if d == 0:
	device = 'cpu'
else if d == 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

criteria = nn.NLLLoss()

lnrate = input("Enter learning rate ")
lnrate = float(lnrate)
optimizer = optim.Adam(model.classifier.parameters(), lr=lnrate)

eps = input("enter no. of epochs ")
eps = int(eps)
model_ft = train_model(model, criteria ,optimizer, eps, device)

print(model_ft)

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()

torch.save({
            'class_to_idx': model.class_to_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': eps,
            'hidden': hiddenUs
            }, 'checkpoint.pth')

print("model saved to checkpoint.pth")