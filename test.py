import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom Dataset to load images and their true labels
class LabeledImageDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Read the CSV file, specifying the relevant columns
        self.labels_df = pd.read_csv(label_file, usecols=["ImagePath", "Label"])

        # Extract filenames and labels
        self.image_paths = self.labels_df['ImagePath'].values
        self.labels = self.labels_df['Label'].apply(
            lambda x: 1 if x == 'real' else 0).values  # Convert 'real'/'fake' to 1/0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])  # Construct full path
        image = Image.open(image_path).convert('RGB')  # Open image as RGB

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()  # Apply threshold

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# Define the data transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset with the new structure
test_dataset = LabeledImageDataset(
    image_folder=".",
    label_file="sample_submission.csv",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load and evaluate the model (code remains the same as earlier)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("binary_classifier.pth", weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Evaluate the model
predictions, true_labels = evaluate(model, test_loader, device)

# Compute performance metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Compute confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Fake', 'Real'])
plt.yticks(tick_marks, ['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="red")
plt.tight_layout()
plt.show()

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(true_labels, predictions)

plt.figure()
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid()
plt.show()

