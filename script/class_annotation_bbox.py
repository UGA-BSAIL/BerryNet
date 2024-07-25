import os
import torch
import cv2
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn

# define the maturity classifier model
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

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
model = SimpleCNN().to(device)
checkpoint = torch.load('/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/maturiety_cnn_classfication/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# define the image transformer
transformer = transforms.Compose([
    transforms.Resize((64, 64)),  # resize the image to the same size as the training images
    # transforms.ToTensor(),
])

# load the image and annotation
image_folder = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230424/train/images'
annotation_folder = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230424/train/labels'

for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # read the image and get its dimensions
            img = read_image(img_path)
            h, w = img.shape[1], img.shape[2]
            updated_lines = []

            # iterate through each bounding box annotation
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                _, x_center, y_center, width, height = map(float, parts)
                
                # convert normalized coordinates to pixel coordinates
                x_center, y_center, width, height = x_center*w, y_center*h, width*w, height*h
                x1, y1 = int(x_center - width//2), int(y_center - height//2)
                x2, y2 = int(x_center + width//2), int(y_center + height//2)

                # extract the region of interest (ROI) from the image
                roi = img[:, y1:y2, x1:x2].float()/255
                roi = transformer(roi).unsqueeze(0).to(device)

                # perform inference on the ROI
                outputs = model(roi)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.item()
                
                # update the class label in the annotation
                parts[0] = str(predicted)
                updated_line = ' '.join(parts)
                updated_lines.append(updated_line)

            # save the updated annotation
            with open(txt_path, 'w') as f:
                for line in updated_lines:
                    f.write("%s\n" % line)
