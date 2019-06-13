import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def imagesets(data_dir):
    return {
        "train": datasets.ImageFolder(data_dir + "train", transform=transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        "valid": datasets.ImageFolder(data_dir + "valid", transform=transforms.Compose([
            transforms.Resize(250),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    }

def dataloaders(imagesets):
    # Load images using ImageFolder + DataLoader
    return {
        "train": torch.utils.data.DataLoader(imagesets["train"], batch_size=32, shuffle=True),
        "valid": torch.utils.data.DataLoader(imagesets["valid"], batch_size=32, shuffle=True)
    }

def load_and_proces_image(filename):
    # Load
    image = Image.open(filename)

    # Resize
    image = image.resize((256, 256))
    
    # Crop
    cx = round((image.width - 224) / 2)
    cy = round((image.height - 224) / 2)
    image = image.crop((
        cx,
        cy,
        cx + 224,
        cx + 224
    ))
    
    np_image = np.array(image)
    np_image = (np_image / 255 - [0.229, 0.224, 0.225]) / [0.485, 0.456, 0.406]
    
    # Change order of np_array columns from [x,y,rgb] to [rgb,x,y]
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.Tensor([np_image])
