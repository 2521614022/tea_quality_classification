import torch
from torch import nn
import torchvision
from PIL import Image
import numpy as np


def main():
    model_straightness = torchvision.models.resnet18(pretrained=False)
    model_straightness.fc = nn.Linear(model_straightness.fc.in_features, 3)
    model_straightness.load_state_dict(torch.load("../models/finetune_resnet_straightness.pth"))
    model_straightness.eval()
    
    model_smoothness = torchvision.models.resnet18(pretrained=False)
    model_smoothness.fc = nn.Linear(model_smoothness.fc.in_features, 3)
    model_smoothness.load_state_dict(torch.load("../models/finetune_resnet_smoothness.pth"))
    model_smoothness.eval()
    
    model_tenderness = torchvision.models.resnet18(pretrained=False)
    model_tenderness.fc = nn.Linear(model_tenderness.fc.in_features, 3)
    model_tenderness.load_state_dict(torch.load("../models/finetune_resnet_tenderness.pth"))
    model_tenderness.eval()

    model_moisture = torchvision.models.resnet18(pretrained=False)
    model_moisture.fc = nn.Linear(model_moisture.fc.in_features, 2)
    model_moisture.load_state_dict(torch.load("../models/finetune_resnet_moisture.pth"))
    model_moisture.eval()
    
    model_fragmentation = torchvision.models.resnet18(pretrained=False)
    model_fragmentation.fc = nn.Linear(model_fragmentation.fc.in_features, 3)
    model_fragmentation.load_state_dict(torch.load("../models/finetune_resnet_fragmentation.pth"))
    model_fragmentation.eval()
    
    model_greenness = torchvision.models.resnet18(pretrained=False)
    model_greenness.fc = nn.Linear(model_greenness.fc.in_features, 2)
    model_greenness.load_state_dict(torch.load("../models/finetune_resnet_greenness.pth"))
    model_greenness.eval()
    
    model_flatness = torchvision.models.resnet18(pretrained=False)
    model_flatness.fc = nn.Linear(model_flatness.fc.in_features, 2)
    model_flatness.load_state_dict(torch.load("../models/finetune_resnet_flatness.pth"))
    model_flatness.eval()
    
    model_uniformity = torchvision.models.resnet18(pretrained=False)
    model_uniformity.fc = nn.Linear(model_uniformity.fc.in_features, 3)
    model_uniformity.load_state_dict(torch.load("../models/finetune_resnet_uniformity.pth"))
    model_uniformity.eval()
    
    with torch.no_grad():
        # Create a converter to convert PIL images to Tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485,0.456,0.406],
                                             [0.229,0.224,0.225])
        ])
        test_image = transform(Image.open("../test_image.jpg"))
        test_image = torch.unsqueeze(test_image, dim=0)
        
        predicted = []
        
        output = model_straightness(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_smoothness(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_tenderness(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_moisture(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_fragmentation(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_greenness(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_flatness(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        output = model_uniformity(test_image)
        predicted.append(output.argmax(dim=1, keepdim=False))
        
        labels = [
            ['near straight', 'sharp', 'straight'],
            ['approach smooth', 'near smooth', 'smooth'],
            ['default', 'with buds', 'with some buds'],
            ['bloom', 'near bloom'],
            ['approach even', 'even', 'near even'],
            ['green', 'near green'],
            ['flat', 'near flat'],
            ['approach even', 'even', 'near even']
        ]
        
        print(f'Straightness: {labels[0][predicted[0].item()]}')
        print(f'Smoothness: {labels[1][predicted[1].item()]}')
        print(f'Tenderness: {labels[2][predicted[2].item()]}')
        print(f'Moisture degree: {labels[3][predicted[3].item()]}')
        print(f'Integral fragmentation: {labels[4][predicted[4].item()]}')
        print(f'Greenness: {labels[5][predicted[5].item()]}')
        print(f'Flatness: {labels[6][predicted[6].item()]}')
        print(f'Color Uniformity: {labels[7][predicted[7].item()]}')
        

if __name__ == "__main__":
    main()