import cv2, sys
sys.path.append("/blue/lift-phenomics/zhengkun.li/yolov8")
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
import os


def density_map_generation(image, boxes, scale=0.25, sigma=15):

    img_height, img_width = image.shape[:2]

    # Create a black image
    img_density = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw white circles at the location of the points
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(img_density, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        # Increment the pixel values within the bounding box region
        # img_density[int(y1):int(y2), int(x1):int(x2)] += 1

    # # Reverse the grayscale image
    img_density = np.max(img_density) - img_density

    # Apply a Gaussian filter to the image to generate the density map
    img_density = cv2.GaussianBlur(img_density, (0, 0), sigmaX=10, sigmaY=10)

    # Normalize the image to 0-255 range and convert to uint8 type
    img_density = cv2.normalize(img_density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply a color map to the grayscale image
    colormap = plt.get_cmap('jet')
    img_color = colormap(img_density)[:, :, 0:3] * 255
    img_color = img_color.astype(np.uint8)  # Convert to uint8

    # Blending images
    density_map_blending = cv2.addWeighted(image, 1, img_color, 0.3, 0, dtype=cv2.CV_8U)

    return img_color, density_map_blending




# Load the YOLOv8 model
model = YOLO('/blue/lift-phenomics/zhengkun.li/blueberry_project/weights/model_blueberry_ASABE/yolov8x_1280.pt')

# Open the video file
video_path = "/blue/lift-phenomics/zhengkun.li/sam/blueberry_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the annotated video
output_path = "/blue/lift-phenomics/zhengkun.li/sam/sam_prediction_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


sam_checkpoint = "/blue/lift-phenomics/zhengkun.li/sam/segment-anything/check_point/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Define a list of colors for the masks
colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, max_det = 3000)

        boxes_list = []
        classes_list = []
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_id = result.boxes.cls.long().tolist()
            boxes_list.append(boxes.xyxy.tolist())
            classes_list.append(class_id)
        bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
        class_id = [class_id for classes in classes_list for class_id in classes]
        input_boxes = torch.tensor(bbox, device=model.device)

        predictor = SamPredictor(sam)
        predictor.set_image(frame)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Create a blank mask overlay with the same shape as the input image
        mask_overlay = np.zeros_like(frame)
        
                # Draw bounding boxes on the image
        # for box, class_id in zip(yolov8_boxex, yolov8_class_id):
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Overlay each mask with a different color
        for i, mask in enumerate(masks):
            binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
            
            # Find the contours of the mask
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the mask overlay with a color
            cv2.drawContours(mask_overlay, contours, -1, colors[i % len(colors)], -1)

        # Blend the mask overlay with the original image
        mask_overlay_alpha = cv2.addWeighted(frame, 1, mask_overlay, 0.3, 0)


        # Write the annotated frame to the output video
        output_video.write(mask_overlay_alpha)

        print("-----------")

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, output video writer, and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()
