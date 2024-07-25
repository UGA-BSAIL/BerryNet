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

# Perform morphological processing
kernel1 = np.ones((10, 10), np.uint8)
img_density = cv2.morphologyEx(img_density, cv2.MORPH_CLOSE, kernel1)
img_density = cv2.morphologyEx(img_density, cv2.MORPH_CLOSE, kernel1)

kernel2 = np.ones((10, 10), np.uint8)
img_density = cv2.morphologyEx(img_density, cv2.MORPH_OPEN, kernel2)

# Remove the counter area < 500 pixels
contours, _ = cv2.findContours(img_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_filter = 3000


for contour in contours:
    area = cv2.contourArea(contour)
    if area > contour_filter:
        # cv2.drawContours(img_org, [contour], -1, 0, -1)

        min_rect = cv2.minAreaRect(contour)
        contour_box = cv2.boxPoints(min_rect)
        contour_box = np.intp(contour_box)
        cv2.drawContours(img_org, [contour_box], 0, (0, 0, 255), 2)

        # Check if the YOLO bounding box is within the area of the contour's minimum bounding rectangle
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmin >= contour_box[:, 0].min() and ymin >= contour_box[:, 1].min() \
                    and xmax <= contour_box[:, 0].max() and ymax <= contour_box[:, 1].max():
                # The YOLO bounding box is within the area of the contour, draw it in green
                cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)



# # Apply a Gaussian filter to the image to generate the density map
# img_density = cv2.GaussianBlur(img_density, (0,0), sigmaX=10, sigmaY=10)

# # Normalize the image to 0-255 range and convert to uint8 type
# img_density = cv2.normalize(img_density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the grayscale density map as an image
# cv2.imwrite('density_map_cluster.png', img_density)
cv2.imwrite('density_map_cluster.png', img_org)