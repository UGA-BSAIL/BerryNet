import numpy as np
import sys
import cv2
import os

sys.path.append("/blue/lift-phenomics/zhengkun.li/yolov8")
from ultralytics import YOLO

import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


image_folder = "/blue/lift-phenomics/zhengkun.li/blueberry_project/data/Xueping_2D/all_images/test" 
model_path = "/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/segmentation/yolov8x_seg-C2f-Faster/exp2/weights/best.pt"
save_path = "/blue/lift-phenomics/zhengkun.li/blueberry_project/data/Xueping_2D/all_images/test_analysis"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the YOLOv8 model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path, task='segmentation')

# Class IDs, names, and colors with shortened names
class_info = {0: {'name': 'im', 'color': (0, 255, 255)},  # Yellow for 'immature'
              1: {'name': 'ma', 'color': (255, 0, 0)},    # Blue for 'mature'
              2: {'name': 'se', 'color': (0, 0, 255)}}    # Red for 'semi_mature'

# Initialize a DataFrame to store the results
results_saving = pd.DataFrame(columns=['file_name', 'num_berries', 'num_immature', 'num_mature', 'num_semi_mature', 'perc_immature', 'perc_mature', 'perc_semi_mature', 'normalized_average_distance', 'cluster_density', 'fruit_density_in_image'])

def process_patch(patch, counts):
    preds = model.predict(patch)
    for r in preds:
        for box in r.boxes:
            class_id = box.cls
            if class_id == 0: counts['immature'] += 1
            elif class_id == 1: counts['mature'] += 1
            elif class_id == 2: counts['semi_mature'] += 1
    return counts


# Define a function to calculate the centroids of the masks
def calculate_centroids(masks):
    centroids = []
    for mask in masks:
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append([cX, cY])
    return np.array(centroids)


# Get the list of all files in the image folder
all_files = os.listdir(image_folder)

# Sort the files
sorted_files = sorted(all_files)

sampling_interval = 4

# Process each image
for file_index, file_name in enumerate(sorted_files):

    # Process one image every sampling_interval images
    if file_index % sampling_interval != 0:
        continue

    print("====================================")
    print(f'Processing {file_name}...')

    img_path = os.path.join(image_folder, file_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Adjust the image size to be divisible by split_scale
    split_scale = 3
    h, w = img.shape[:2]
    adjusted_h = h - (h % split_scale)
    adjusted_w = w - (w % split_scale)
    img = img[:adjusted_h, :adjusted_w]

    # Split the image into split_scale x split_scale patches
    h, w, _ = img.shape
    patch_height, patch_width = h // split_scale, w // split_scale
    counts = {'immature': 0, 'mature': 0, 'semi_mature': 0}

    patches = [img[y:y+patch_height, x:x+patch_width] for y in range(0, adjusted_h, patch_height) for x in range(0, adjusted_w, patch_width)]

    mask_img = np.zeros((h, w, 3), dtype=np.uint8)  # Black image for drawing colored masks
    all_masks = []  # List to store all masks
    all_class_id = []  # List to store all class-id

    # Process each patch
    for i, patch in enumerate(patches):
        # Segment with YOLOv8
        results = model.predict(patch, conf=0.3)

        # Check if results are not empty
        if results is None or len(results) == 0:
            continue
        
        # Calculate the offset for the current patch
        row_index, col_index = i // split_scale, i % split_scale
        x_offset, y_offset = col_index * patch_width, row_index * patch_height


        # Draw masks and class text
        for result in results:

            # Check if masks are not None
            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):

                mask_points = np.int32([mask])
                # Adjust mask coordinates to the original image scale
                mask_points[:, :, 0] += x_offset
                mask_points[:, :, 1] += y_offset
                all_masks.append(mask_points)
                all_class_id.append(int(box.cls[0]))

                points = np.int32([mask])
                class_id = int(box.cls[0])
                color = class_info.get(class_id, {'color': (255, 255, 255)})['color']  # White for unknown classes
                # cv2.polylines(patch, [points], isClosed=True, color=color, thickness=1)

                # Draw the mask on the temporary image
                cv2.fillPoly(mask_img, [mask_points], color)  # Draw mask on black image

    # draw masks on the original image
    for mask, class_id in zip(all_masks, all_class_id):
        color = class_info.get(class_id, {'color': (255, 255, 255)})['color']  # White for unknown classes
        cv2.polylines(img, [mask], isClosed=True, color=color, thickness=1)

    # Calculate the centroids of the masks
    centroids = calculate_centroids(all_masks)
    for c in centroids:
        cv2.circle(img, tuple(c), 3, (255, 255, 255), -1)
        cv2.circle(mask_img, tuple(c), 3, (255, 255, 255), -1)

    for i in range(split_scale):
        for j in range(split_scale):
            patch = img[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            counts = process_patch(patch, counts)

    total_berries = sum(counts.values())
    perc_immature = (counts['immature'] / total_berries) * 100 if total_berries > 0 else 0
    perc_mature = (counts['mature'] / total_berries) * 100 if total_berries > 0 else 0
    perc_semi_mature = (counts['semi_mature'] / total_berries) * 100 if total_berries > 0 else 0

    # Find the nearest neighbor for each centroid
    nbrs = NearestNeighbors(n_neighbors=2).fit(centroids)
    distances, _ = nbrs.kneighbors(centroids)
    nearest_neighbor_distances = distances[:, 1]

    # Calculate the average distance to the nearest neighbor
    average_distance = np.mean(nearest_neighbor_distances)

    # calculate the average size of the fruit
    average_size = np.mean([cv2.contourArea(mask) for mask in all_masks])

    # normalize the average distance by the average size
    normalized_average_distance = average_distance / (2 * np.sqrt(average_size/np.pi)) if average_size else 0
    
    cluster_density = average_size/(np.pi * normalized_average_distance**2)

    #claculate the density of the fruit in the image
    fruit_density_in_image = sum(cv2.contourArea(mask) for mask in all_masks) / (h * w)




    new_row = pd.DataFrame([{
        'file_name': file_name,
        'num_berries': total_berries,
        'num_immature': counts['immature'],
        'num_mature': counts['mature'],
        'num_semi_mature': counts['semi_mature'],
        'perc_immature': perc_immature,
        'perc_mature': perc_mature,
        'perc_semi_mature': perc_semi_mature,
        'normalized_average_distance': normalized_average_distance,
        'cluster_density': cluster_density,
        'fruit_density_in_image': fruit_density_in_image
    }])

    results_saving  = pd.concat([results_saving , new_row], ignore_index=True)


    cv2.putText(img, f'immature: {counts["immature"]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 1)
    cv2.putText(img, f'mature: {counts["mature"]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 1)
    cv2.putText(img, f'semi_mature: {counts["semi_mature"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)

    # Save the image with masks
    cv2.imwrite(f'{save_path}/segment_{file_name}', img)
    cv2.imwrite(f'{save_path}/mask_image{file_name}', mask_img)

# Save results
results_saving .to_csv(f'{save_path}/segmentation_maturiety_results.csv', index=False)