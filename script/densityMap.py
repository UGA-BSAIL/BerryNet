import cv2
import numpy as np
import matplotlib.pyplot as plt


def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box


# Load image and text file with different names
img_filename = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_total/test/images/top_FLR-12-89_JPG.rf.5fa4b0013fc4f42d70236780f11c1264.jpg'
text_filename = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_total/predict/labels/top_FLR-12-89_JPG.rf.5fa4b0013fc4f42d70236780f11c1264.txt'

img_org = cv2.imread(img_filename)
img_height, img_width = img_org.shape[:2]

# Load the YOLO detection results from a text file
with open(text_filename, 'r') as f:
    detections = f.readlines()

# Convert the YOLO format from () (yolov8 default formate) to (xmin, ymin, xmax, ymax)
boxes = []
for detection in detections:
    class_id, x, y, w, h = map(float, detection.strip().split())
    xmin = int((x - w / 2) * img_width)
    ymin = int((y - h / 2) * img_height)
    xmax = int((x + w / 2) * img_width)
    ymax = int((y + h / 2) * img_height)
    boxes.append((xmin, ymin, xmax, ymax))


# Create a black image
img_density = np.zeros((img_height, img_width), dtype=np.uint8)

# Draw white circles at the location of the points
for box in boxes:
    cv2.rectangle(img_density, (box[0], box[1]), (box[2], box[3]), 255, -1)

# # Reverse the grayscale image
img_density = np.max(img_density) - img_density

# Apply a Gaussian filter to the image to generate the density map
img_density = cv2.GaussianBlur(img_density, (0,0), sigmaX=10, sigmaY=10)

# Normalize the image to 0-255 range and convert to uint8 type
img_density = cv2.normalize(img_density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the grayscale density map as an image
cv2.imwrite('density_map_grayscale.png', img_density)

# Apply a color map to the grayscale image
colormap = plt.get_cmap('jet')
img_color = colormap(img_density)[:,:,0:3] * 255

# Save the color density map as an image
cv2.imwrite('density_map_color.png', img_color)

# Blending images
# density_map_blending = cv2.addWeighted(img_org, 0.7, img_color, 0.3, 0)
density_map_blending = cv2.addWeighted(img_org, 1, img_color, 0.3, 0, dtype=cv2.CV_8U)
# density_map_blending_rgba = cv2.addWeighted(img_org, 1, img_color_rgba, 0.3, 0, dtype=cv2.CV_8U)
cv2.imwrite('density_map_blending.png', density_map_blending)





