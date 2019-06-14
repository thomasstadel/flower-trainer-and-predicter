import json
import os
import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

def cat_to_name(filename = 'cat_to_name.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def available_models():
    return ("alexnet", "vgg11", "vgg13", "vgg16", "vgg19", "densenet121", "densenet169", "densenet161", "densenet201")

class pretrained_model:

    def __init__(self, model, hidden_units = 512, learning_rate = 0.001, class_to_idx = {}):
        # Settings
        self.device = "cpu"
        self.modelname = model
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.class_to_idx = class_to_idx

        if model not in available_models():
            print(f"{model} isn't an available model")
            exit()

        # Getting pretrained model
        self.model = eval("models." + model + "(pretrained=True)")
        
        # In-features to classifier
        in_features = 0
        if hasattr(self.model.classifier, "in_features"):
            in_features = self.model.classifier.in_features
        else:
            for cls in self.model.classifier:
                if hasattr(cls, "in_features"):
                    in_features = cls.in_features
                    break
        if in_features == 0:
            print("Ohh no, something went wrong - couldn't figure out the number of in_features to the classifier")
            exit()

        # Freeze parameters
        for param in self.model.parameters():
            param.required_grad = False

        # Our own classifier, in_features=from model above, out_features=len(cat_to_name)
        classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden_units)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(hidden_units, len(class_to_idx))),
            ("output", nn.LogSoftmax(dim=1))
        ]))
        self.model.classifier = classifier

        # Loss function
        self.criterion = nn.NLLLoss()

        # Optimzer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def gpu(self):
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.model.to(self.device)
            return True
        else:
            return False

    def cpu(self):
        self.device = "cpu"
        self.model.to(self.device)
        return True

    def save(self, filename):
        checkpoint = {
            "model": self.modelname,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "state_dict": self.model.state_dict(),
            "class_to_idx": self.class_to_idx
        }
        torch.save(checkpoint, filename)
        return True

    def train(self, images, labels):
        # Move to device
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward
        output = self.model(images)
        
        # Loss
        loss = self.criterion(output, labels)
        loss.backward()
        
        # Optimze
        self.optimizer.step()

        return loss.item()

    def validate(self, images, labels):
        # Evaluate
        self.model.eval()

        with torch.no_grad():
            # Move to device
            images, labels = images.to(self.device), labels.to(self.device)
                
            # Forward
            output = self.model(images)
                
            # Loss
            loss = self.criterion(output, labels)
            
            # Find top class
            ps = torch.exp(output)
            top_label, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

        # Back to training mode
        self.model.train()

        return loss.item(), accuracy

    def predict(self, image, top_k = 5):
        # Evaluate
        self.model.eval()

        with torch.no_grad():
            # Move to device
            image = image.to(self.device)
                
            # Forward
            output = self.model(image)
                
            # Find top class
            ps = torch.exp(output)
            top_probs, top_class = ps.topk(top_k, dim=1)

        # Back to training mode
        self.model.train()

        # To numpy
        top_probs = top_probs.cpu().numpy().squeeze()
        top_class = top_class.cpu().numpy().squeeze()

        return top_probs, [list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(x)] for x in top_class]


class pretrained_model_load(pretrained_model):

    def __init__(self, filename, gpu):

        # Check if file exists
        if os.path.isfile(filename) == False:
            print(f"{filename} doesn't exist")
            exit()

        # Load checkpoint
        checkpoint = torch.load(filename, map_location="cuda:0" if gpu else "cpu")

        # Call __init__ of the extended class
        super().__init__(checkpoint["model"], checkpoint["hidden_units"], checkpoint["learning_rate"], checkpoint["class_to_idx"])

        # Device
        if gpu:
            self.gpu()

        # Load state dict to model
        self.model.load_state_dict(checkpoint["state_dict"])
