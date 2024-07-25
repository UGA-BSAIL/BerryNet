import os
import torch
import cv2
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import argparse, sys, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import time

def get_normalized_mask_coordinates(mask, width, height):
    """
    Normalize bounding box coordinates.
    mask: binary mask of the object
    width, height: dimensions of the image
    returns: normalized center coordinates and dimensions of the bounding box
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    norm_width = (xmax - xmin) / width
    norm_height = (ymax - ymin) / height

    return x_center, y_center, norm_width, norm_height

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network model.
    """
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

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained CNN model and its weights
model = SimpleCNN().to(device)
checkpoint = torch.load('/path/to/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define image transformations
transformer = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Load the SAM model for segmentation
sam_checkpoint = "/path/to/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Paths to image and annotation folders
image_folder = '/path/to/image_folder'
annotation_folder = '/path/to/annotation_folder'

# Iterate over each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # Read the image
            img = read_image(img_path)
            h, w = img.shape[1], img.shape[2]
            updated_lines = []

            gt_bboxes = []
            class_list = []
            for line in lines:
                parts = line.strip().split()
                _, x_center, y_center, width, height = map(float, parts)
                
                # Convert normalized coordinates to pixel values
                x_center, y_center, width, height = x_center*w, y_center*h, width*w, height*h
                x1, y1 = int(x_center - width//2), int(y_center - height//2)
                x2, y2 = int(x_center + width//2), int(y_center + height//2)

                gt_bboxes.append([x1, y1, x2, y2])

                # Extract ROI (Region of Interest)
                roi = img[:, y1:y2, x1:x2].float()/255
                roi = transformer(roi).unsqueeze(0).to(device)

                # Perform classification on the ROI
                outputs = model(roi)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.item()
                
                class_list.append(predicted)
                
            # Perform segmentation using SAM
            start_time = time.time()
            img = cv2.imread(img_path)
            predictor = SamPredictor(sam)
            predictor.set_image(img)

            input_boxes = torch.tensor(gt_bboxes, device=device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Calculate FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time
            print(f"FPS: {fps}")

            for i, mask in enumerate(masks):
                binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) != 0:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Normalize the mask points
                    mask_points_normalized = largest_contour / np.array([w, h])
                    mask_points_normalized = mask_points_normalized.reshape(-1, 2)
                    mask_points_str_list = ' '.join(mask_points_normalized.flatten().astype(str).tolist())

                    new_line = str(class_list[i]) + ' ' + mask_points_str_list
                    updated_lines.append(new_line)

            # Write updated annotations to the file
            with open(txt_path, 'w') as f:
                for line in updated_lines:
                    f.write("%s\n" % line)
