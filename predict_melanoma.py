import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.classifier[1].in_features, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

def load_models(model_paths):
    models = []
    for path in model_paths:
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_tta_transforms():
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

def predict(models, image_path, transform, tta_transforms, weights):
    image = Image.open(image_path).convert('RGB')

    preds = []
    with torch.no_grad():
        for model, weight in zip(models, weights):
            model_preds = []
            for tta_transform in tta_transforms:
                tta_image = tta_transform(image).unsqueeze(0).to(device)
                output = model(tta_image)
                model_preds.append(output.cpu().numpy())
            avg_model_pred = np.mean(model_preds, axis=0)
            preds.append(avg_model_pred * weight)

    final_pred = np.sum(preds) / np.sum(weights)
    final_pred = np.round(final_pred).item()

    return final_pred

def main(image_path):
    model_paths = [
        'models/model_fold_1.pth',
        'models/model_fold_2.pth',
        'models/model_fold_3.pth',
        'models/model_fold_4.pth',
        'models/model_fold_5.pth'
    ]

    models = load_models(model_paths)
    transform = get_image_transforms()
    tta_transforms = get_tta_transforms()
    weights = [1, 1, 1, 1, 1]

    prediction = predict(models, image_path, transform, tta_transforms, weights)
    if prediction == 1:
        print("The image is predicted to be melanoma.")
    else:
        print("The image is predicted to be non-melanoma.")

if __name__ == '__main__':
    image_path = 'BCC2.jpg'
    main(image_path)
