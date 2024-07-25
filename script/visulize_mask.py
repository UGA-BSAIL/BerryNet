from PIL import Image,ImageDraw 
import os
from pathlib import Path 
from shutil import copyfile 
from tqdm import tqdm
import numpy as np 
from pathlib import Path

def get_labels_polys(img_path,gt_path):
    img = Image.open(img_path)
    w,h = img.size  
    with open(gt_path, 'r') as fl:
        lines = [x.rstrip() for x in fl.readlines()]
    str_data = [x.split(' ') for x in lines]
    relative_polys = [[float(x) for x in arr[1:]] for arr in str_data]
    labels = [int(arr[0]) for arr in str_data]
    polys = [ [x*w if i%2==0 else x*h  for i,x in enumerate(arr)]  for arr in relative_polys]
    return labels,polys

def plot_polys(image, labels, polys):
    image_result = image.copy()
    draw = ImageDraw.Draw(image_result)
    color_map = {0: 'green', 1: 'blue', 2: 'red'}

    for label, poly in zip(labels, polys):
        draw.polygon(poly, outline=color_map[label])
        # draw.polygon(poly, outline=color_map[label], fill=color_map[label])
    return image_result

# Root path
root_path = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/3'

# Output directory for processed images
output_dir = Path(root_path) / 'processed_images'
output_dir.mkdir(exist_ok=True)

# Path setup
data_root = Path(root_path)
val_imgs = [str(x) for x in (data_root / 'images').rglob("*.jpg") if 'checkpoint' not in str(x)]

# Process each image
for img_path in val_imgs:
    gt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

    labels, polys = get_labels_polys(img_path, gt_path)
    processed_image = plot_polys(Image.open(img_path), labels, polys)

    # Save the processed image
    output_img_path = output_dir / Path(img_path).name
    processed_image.save(output_img_path)

print("Processing complete. Images saved in:", output_dir)




















# from PIL import Image, ImageDraw
# import numpy as np
# from pathlib import Path

# def get_labels_polys(img_path, gt_path):
#     img = Image.open(img_path)
#     w, h = img.size
#     with open(gt_path, 'r') as fl:
#         lines = [x.rstrip() for x in fl.readlines()]
#     str_data = [x.split(' ') for x in lines]
#     relative_polys = [[float(x) for x in arr[1:]] for arr in str_data]
#     labels = [int(arr[0]) for arr in str_data]
#     polys = [[x*w if i%2==0 else x*h for i, x in enumerate(arr)] for arr in relative_polys]
#     return labels, polys

# def plot_polys(image, labels, polys):
#     image_result = image.copy()
#     draw = ImageDraw.Draw(image_result)
#     color_map = {0: 'green', 1: 'blue', 2: 'red'}

#     for label, poly in zip(labels, polys):
#         draw.polygon(poly, outline=color_map[label])
#     return image_result

# # Root path
# root_path = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/3'

# # Output directory for processed images
# output_dir = Path(root_path) / 'processed_images'
# output_dir.mkdir(exist_ok=True)

# # Path setup
# data_root = Path(root_path)
# image_paths = list((data_root / 'images').glob("*.jpg"))

# # Process each image
# for img_path in image_paths:
#     gt_path = img_path.with_name(img_path.stem + '.txt').with_suffix('.txt').resolve()

#     labels, polys = get_labels_polys(str(img_path), str(gt_path))
#     processed_image = plot_polys(Image.open(img_path), labels, polys)

#     # Save the processed image
#     output_img_path = output_dir / img_path.name
#     processed_image.save(output_img_path)

# print("Processing complete. Images saved in:", output_dir)