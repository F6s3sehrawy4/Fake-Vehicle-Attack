import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Paths to your datasets
unedited_path = "dataset/train/real"
edited_path = "dataset/train/fake"

# Custom Dataset for Edited and Unedited Scenes
class SceneDataset(Dataset):
    def __init__(self, unedited_path, edited_path, transform=None):
        self.unedited_images = [os.path.join(unedited_path, img) for img in os.listdir(unedited_path)]
        self.edited_images = [os.path.join(edited_path, img) for img in os.listdir(edited_path)]
        self.images = self.unedited_images + self.edited_images
        self.labels = [0] * len(self.unedited_images) + [1] * len(self.edited_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((450, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
dataset = SceneDataset(unedited_path=unedited_path, edited_path=edited_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Ensure output is 1x1 per channel
            nn.Conv2d(512, 1, kernel_size=1),  # Reduce channels to 1
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1)  # Flatten to (batch_size, 1)

# Initialize the discriminator and set up optimizer and loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator().to(device)
#criterion = nn.BCELoss()
#optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("Training started...")


# Training loop
num_epochs = 10
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

for epoch in range(num_epochs):
    discriminator.train()
    epoch_loss = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

        # Forward pass
        outputs = discriminator(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the batch loss
        epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Step scheduler
    scheduler.step(avg_loss)
