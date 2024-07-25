import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn

# Define a simple convolutional neural network (CNN) model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = SimpleCNN().to(device)
checkpoint = torch.load('weight\maturity-classifier\best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create a data loader
dataset = datasets.ImageFolder('.dataset/Blueberry_maturity/valid', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Perform inference and save misclassified images
for i, (inputs, labels) in enumerate(dataloader):
    inputs, labels = inputs.to(device), labels.item()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.item()

    # If the prediction is incorrect, save the image
    if predicted != labels:
        label_name = dataset.classes[labels]
        pred_name = dataset.classes[predicted]
        folder_path = f'error_images/{label_name}_as_{pred_name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Get the original image path
        image_path = dataset.imgs[i][0]
        image = Image.open(image_path)

        # Save the image
        error_image_path = os.path.join(folder_path, os.path.basename(image_path))
        image.save(error_image_path)

print("Finished saving incorrectly classified images.")
