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


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoints(path):
    lcheckpoint = torch.load(path)
    
    model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
            param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    model.load_state_dict(lcheckpoint['model_state_dict'])
    model.class_to_idx = lcheckpoint['class_to_idx']
        
    return model

premodel = load_checkpoints('checkpoint.pth')
print("model is loaded")

print(premodel.classifier)

classname = input("enter file name where image classes are stored ") 
with open(classname, 'r') as f:
    cat_to_name = json.load(f)

def process_image(img_path):

    image = Image.open(img_path)
    
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    
#     print(image)
    
    left_margin = (image.size[0]-224)/2
    bottom_margin = (image.size[0]-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    image = image.crop((left_margin, bottom_margin, right_margin,    
                   top_margin))
    
    image = np.array(image)/255
    
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


image_path = 'flowers/test/1/image_06743.jpg'
img = process_image(image_path)
# imshow(img)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path)
#     imshow(image)
    
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image.unsqueeze_(0)
    
    ps = torch.exp(model.forward(image))
    
#     print(prob)
#     print(classes)
    
    top_probs, top_labs = ps.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers


def plot_solution(image_path, model, k):
    # Set up plot
#     plt.figure(figsize = (6,10))
#     ax = plt.subplot(2,1,1)    # Set up title
    flower_num = image_path.split('/')[2]
    print(flower_num)
    title_ = cat_to_name[flower_num]    # Plot flower
    img = process_image(image_path)
#     imshow(img, ax, title = title_)   # Make prediction
    probs, labs, flowers = predict(image_path, model, k)     # Plot bar chart
#     plt.subplot(2,1,2)
#     sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
#     plt.show()
    return probs, labs, flowers
    

k = input("Enter the value of k parameter in topk function ")
k = int(k)
image_path = input("Enter image path ")
if len(image_path) == 0:
	image_path = 'flowers/test/28/image_05253.jpg'
	
print(plot_solution(image_path, premodel, k))