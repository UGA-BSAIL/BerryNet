import os
import cv2
import numpy as np
from sklearn import linear_model
import csv


def normalize_rgb(image):
    # Convert image to float32 for division operations
    image_float = image.astype(np.float32)
    
    # Calculate the norm for each pixel
    norm = np.sqrt(np.sum(image_float**2, axis=2, keepdims=True))
    
    # Avoid division by zero
    norm[norm == 0] = 1

    # Normalize each channel
    normalized = image_float / norm
    
    return normalized

def normalize_lab(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Separate the LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Normalize the L channel
    l_channel_normalized = cv2.normalize(l_channel.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # Merge the normalized L channel with the A and B channels
    lab_image_normalized = cv2.merge((l_channel_normalized, a_channel, b_channel))
    
    # Convert LAB image back to BGR color space
    normalized_image = cv2.cvtColor(lab_image_normalized, cv2.COLOR_LAB2BGR)
    
    return normalized_image


# Folder paths
image_folder = "/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230509/test/images"
annotation_folder = "/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230509/test/labels"
output_folder = "/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230509/single_fruit/all"

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each image in the folder
for image_file in os.listdir(image_folder):
    print("=========================================")
    # Check if it's a JPG image
    if image_file.endswith('.jpg'):
        
        # Construct paths
        img_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))
        
        # Read image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Check if annotation file exists
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    # Parse the line to get bounding box details
                    parts = line.strip().split()
                    _, x_center, y_center, width, height = map(float, parts)

                    # Convert normalized coordinates to pixel coordinates
                    x_center, y_center, width, height = x_center*w, y_center*h, width*w, height*h
                    x1, y1 = int(x_center - width/2), int(y_center - height/2)
                    x2, y2 = int(x_center + width/2), int(y_center + height/2)

                    # Extract the ROI
                    roi = img[y1:y2, x1:x2]

                    # Save the ROI
                    roi_filename = os.path.join(output_folder, f"{image_file[:15]}_roi_{index}.jpg")
                    # roi = (normalize_rgb(roi)* 255).astype(np.uint8)

                    cv2.imwrite(roi_filename, roi)
                    
                    # Draw bounding box (optional, since you're saving the ROI)
                    # cv2.rectangle(img, (x1, y1))