import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms, models
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MelanomaDetectionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), shear=20, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience, model_path, scheduler=None):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        epoch_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for images, labels in epoch_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            preds = torch.round(outputs)
            running_corrects += torch.sum(preds.squeeze() == labels.data)

            epoch_loader.set_postfix(loss=running_loss / ((epoch_loader.n + 1) * train_loader.batch_size),
                                     acc=running_corrects.double() / ((epoch_loader.n + 1) * train_loader.batch_size))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_accuracies.append(epoch_acc.item())

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_running_loss += loss.item() * images.size(0)
                preds = torch.round(outputs)
                val_running_corrects += torch.sum(preds.squeeze() == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_accuracies.append(val_acc.item())

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        if scheduler:
            scheduler.step(val_loss)

        if epochs_no_improve >= patience:
            print('Early stopping!')
            early_stop = True
            break

    return model, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.round(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.squeeze().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC AUC Score: {roc_auc:.4f}')
    return accuracy, precision, recall, f1, roc_auc

def plot_accuracies(train_accuracies, val_accuracies, fold):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training vs Validation Accuracy - Fold {fold}')
    plt.savefig(f'plots/accuracy_plot_fold_{fold}.png')
    plt.close()

def k_fold_cross_validation(model_class, dataset, k=5, num_epochs=25, patience=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    all_train_accuracies = []
    all_val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=10, pin_memory=True)

        model = model_class().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        model_path = f'models/model_fold_{fold + 1}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model, train_accuracies, val_accuracies = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                              num_epochs, patience, model_path, scheduler)

        val_acc = val_accuracies[-1]
        fold_accuracies.append(val_acc)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)

        plot_accuracies(train_accuracies, val_accuracies, fold + 1)

    avg_accuracy = sum(fold_accuracies) / k
    print(f'Average K-Fold Validation Accuracy: {avg_accuracy:.4f}')
    return fold_accuracies, all_train_accuracies, all_val_accuracies

def ensemble_prediction(models, test_loader):
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds.append(outputs.cpu().numpy())
        all_preds.append(np.concatenate(preds))

    avg_preds = np.mean(all_preds, axis=0)
    final_preds = np.round(avg_preds)

    return final_preds

def main():
    image_dir = './melanoma_dataset'
    csv_file = "./melanoma_dataset/melanoma_data.csv"
    num_epochs = 25
    patience = 5
    batch_size = 32
    num_workers = 10

    image_transforms = get_image_transforms()
    dataset = MelanomaDetectionDataset(csv_file=csv_file, root_dir=image_dir, transform=image_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    fold_accuracies, all_train_accuracies, all_val_accuracies = k_fold_cross_validation(SimpleCNN, dataset, k=5, num_epochs=num_epochs, patience=patience)

    models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        model = SimpleCNN().to(device)
        model_path = f'models/model_fold_{fold + 1}.pth'
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    final_preds = ensemble_prediction(models, test_loader)

    correct = 0
    total = len(test_dataset)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            labels = labels.numpy()
            for j in range(len(labels)):
                if labels[j] == final_preds[i * batch_size + j]:
                    correct += 1

    ensemble_test_accuracy = correct / total
    print(f'Ensemble Test Accuracy: {ensemble_test_accuracy:.4f}')

    # Evaluate each model
    for i, model in enumerate(models):
        print(f'\nEvaluating Model {i + 1}')
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, test_loader)
        print(f'Model {i + 1} Test Accuracy: {accuracy:.4f}')
        print(f'Model {i + 1} Test Precision: {precision:.4f}')
        print(f'Model {i + 1} Test Recall: {recall:.4f}')
        print(f'Model {i + 1} Test F1 Score: {f1:.4f}')
        print(f'Model {i + 1} Test ROC AUC Score: {roc_auc:.4f}')

if __name__ == '__main__':
    main()

