import torch
import torch.nn as nn
import torchvision.models as model


class ExModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        # resnet18       
        self.model = model.resnet18(pretrained=False)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))    
        self.classifier = torch.nn.Linear(512, 2)

        # vgg16
        # self.model = model.vgg16(pretrained=False)
        # self.classifier = torch.nn.Linear(1000, 2)


    def forward(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.model(image).squeeze()
        out = self.classifier(resnet_pred)
        
        return out
    