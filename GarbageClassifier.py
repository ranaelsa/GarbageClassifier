import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# Define dataset class
class CustomImageTextDataset(Dataset):
    def __init__(self, image_dir, transform, tokenizer):
        self.dataset = datasets.ImageFolder(image_dir, transform=transform) 
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  
        image_path, _ = self.dataset.imgs[idx]  # Get image filename

        # Extract text from filename 
        file_name = os.path.basename(image_path)
        text = os.path.splitext(file_name)[0]  # Remove extension
        text = text.replace('_', ' ')  # Replace underscores with spaces
        text = re.sub(r'\d+', '', text)  # Remove numbers

        # Tokenize text using DistilBERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
# Define the model
class ResNetBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBERTClassifier, self).__init__()

        # Image feature extractor
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze layers for transfer learning
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Remove the final layer
        self.resnet.fc = nn.Identity()
        self.resnet_out_dim = 512  # Output feature size

        # Text feature extracter
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_out_dim = self.bert.config.hidden_size  
        
        # Freeze all layers except the last two transformer layers
        for name, param in self.bert.named_parameters():
            if "transformer.layer.4" in name or "transformer.layer.5" in name:
                param.requires_grad = True  
            else:
                param.requires_grad = False  

        # Single FC layer for classification
        self.classifier = nn.Linear(self.resnet_out_dim + self.bert_out_dim, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # Image pathway
        img_features = self.resnet(image)
        img_features = F.normalize(img_features, p=2, dim=1)  # L2 normalization

        # Text pathway
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = text_output[:, 0] 
        text_features = F.normalize(text_features, p=2, dim=1)  # L2 normalization

        # Combine image and text features
        combined_features = torch.cat((img_features, text_features), dim=1)

        # Pass through the classifier
        output = self.classifier(combined_features)
        return output

# Define function to train the model
def train_model(model, dataloaders, optimizer, criterion, device, num_epochs=10, save_path='./best_model.pth'):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Enable training mode
            else:
                model.eval()  # Enable evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for batch in dataloaders[phase]:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, input_ids, attention_mask)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation + optimization during training only
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save the best model directly
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with loss: {best_loss:.4f}")

    print("\nTraining complete. Best validation loss: {:.4f}".format(best_loss))
    return model

# Define function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []  # Store predicted class labels
    all_probs = []  # Store softmax probabilities

    with torch.no_grad():  # No gradient calculation needed
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Get class predictions

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # Store probabilities

    accuracy = (torch.tensor(all_labels) == torch.tensor(all_preds)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'labels': all_labels,   # Return true labels for confusion matrix
        'predictions': all_preds,  # Return predicted class labels for confusion matrix
        'probabilities': all_probs  # Return softmax probabilities for ROC curve
    }

# Define data directories
TRAIN_PATH  = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH    = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH   = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# Define transformations for the images
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load datasets
datasets = {
    "train": CustomImageTextDataset(TRAIN_PATH, transform=transform["train"], tokenizer=tokenizer),
    "val": CustomImageTextDataset(VAL_PATH, transform=transform["val"], tokenizer=tokenizer),
    "test": CustomImageTextDataset(TEST_PATH, transform=transform["test"], tokenizer=tokenizer)
}

# Define data loaders
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=32, shuffle=True, num_workers=2), # Shuffle training only
    "val": DataLoader(datasets["val"], batch_size=32, shuffle=False, num_workers=2),
    "test": DataLoader(datasets["test"], batch_size=32, shuffle=False, num_workers=2),
}

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ResNetBERTClassifier(num_classes=4).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Train the model
model = train_model(model, dataloaders, optimizer, criterion, device, num_epochs=10, save_path='./best_model.pth')

# Load the best model
model.load_state_dict(torch.load('./best_model.pth'))

# Evaluate the model
eval_results = evaluate_model(model, dataloaders['test'], device)

# Print Evaluation Metrics
print("\n--- Evaluation Metrics on Test Set ---")
print(f"Test Accuracy : {eval_results['accuracy']:.4f}")
print(f"Precision     : {eval_results['precision']:.4f}")
print(f"Recall        : {eval_results['recall']:.4f}")
print(f"F1 Score      : {eval_results['f1_score']:.4f}")
