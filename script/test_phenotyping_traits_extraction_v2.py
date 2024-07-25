# from ultralytics import YOLO
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import argparse, sys, os, warnings
sys.path.append("/blue/lift-phenomics/zhengkun.li/yolov8")
from ultralytics import YOLO
warnings.filterwarnings('ignore')
from pathlib import Path


def transformer_opt(opt):
    opt = vars(opt)
    del opt['source']
    del opt['weight']
    return opt

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/weights/model_blueberry_ASABE/yolov8x_1280.pt', help='training model path')
    parser.add_argument('--source', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230424/train/images', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'track'], help='predict mode or track mode')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--show', action="store_true", help='show results if possible')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_conf', action="store_true", help='save results with confidence scores')
    parser.add_argument('--show_labels', action="store_true", default=True, help='show object labels in plots')
    parser.add_argument('--show_conf', action="store_true", default=False, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=3, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action="store_true", help='visualize model features')
    parser.add_argument('--augment', action="store_true", help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action="store_true", help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--retina_masks', action="store_true", help='use high-resolution segmentation masks')
    parser.add_argument('--boxes', action="store_true", default=True, help='Show boxes in segmentation predictions')
    parser.add_argument('--save', action="store_true", default=True, help='save result')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml'], help='tracker type, [botsort.yaml, bytetrack.yaml]')
    parser.add_argument('--max_det', type=int, default=3000, help='maximum number of detections per image')
    
    return parser.parse_known_args()[0]

class YOLOV8(YOLO):
    '''
    weigth:model path
    '''
    def __init__(self, weight='', task=None) -> None:
        super().__init__(weight, task)

def yolov8_detection(model, image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, stream=True)  # generator of Results objects

    boxes_list = []
    classes_list = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        class_id = result.boxes.cls.long().tolist()
        boxes_list.append(boxes.xyxy.tolist())
        classes_list.append(class_id)

    bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
    class_id = [class_id for classes in classes_list for class_id in classes]

    return bbox, class_id, image

def blueberry_cluster(img_org, boxes, masks, contour_area_filter = 3000):


    # Overlay each mask with a different color
    # mask_overlay_all = np.zeros_like(img_org)
    # for i, mask in enumerate(masks):
    #     binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
        
    #     # Find the contours of the mask
    #     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     # Draw contours on the mask overlay with a color
    #     cv2.drawContours(mask_overlay_all, contours, -1, colors[i % len(colors)], -1)


    img_height, img_width = img_org.shape[:2]
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

    contour_filter = contour_area_filter
    num_mature_blueberries_image = 0
    num_immature_blueberries_image = 0
    cluster_num  = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_filter:
            cluster_num  += 1
            # cv2.drawContours(img_org, [contour], -1, 0, -1)

            min_rect = cv2.minAreaRect(contour)
            min_rect_width, min_rect_height = min_rect[1]
            contour_box = cv2.boxPoints(min_rect)
            contour_box = np.intp(contour_box)
            cv2.drawContours(img_org, [contour_box], 0, (0, 0, 255), 2)

            # Check if the YOLO bounding box is within the area of the contour's minimum bounding rectangle
            mask_overlay = np.zeros_like(img_org)
            num_mature_blueberries_cluster = 0
            num_immature_blueberries_cluster = 0
            lower_blueberry = np.array([90, 50, 50])   # Lower threshold for hue, saturation, and value
            upper_blueberry = np.array([130, 255, 255])  # Upper threshold for hue, saturation, and value
            hsv_img = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
            cluster_mask_area = 0
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = box
                if xmin >= contour_box[:, 0].min() and ymin >= contour_box[:, 1].min() \
                        and xmax <= contour_box[:, 0].max() and ymax <= contour_box[:, 1].max():
                    # The YOLO bounding box is within the area of the contour, draw it in green
                    # cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # ----------------------------- mask operation-----------------------#
                    # draw mask
                    binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                    # Find the contours of the mask
                    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cluster_mask_area += cv2.contourArea(contours[0])

                    # Draw contours on the mask overlay with a color
                    cv2.drawContours(mask_overlay, contours, -1, colors[i % len(colors)], -1)

                    # identify the maturaty of each mask
                    # Create a mask image that contains the contour filled in white
                    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

                    # Convert the original image to HSV
                    hsv_image = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
                    # Extract the H channel
                    h_channel = hsv_image[:, :, 0]
                    # Use the mask to get the Hue values within the contour
                    hue_values_within_contour = h_channel[mask == 255]
                    mean_hue = np.mean(hue_values_within_contour)
                    print (mean_hue)

                    # Identify maturity based on mean hue value
                    maturity_text = "Ma"
                    immaturity_text = "Im"
                    if mean_hue > 90 and mean_hue < 115:
                        num_mature_blueberries_cluster += 1
                        # cv2.putText(img_org, maturity_text, (int((xmin+xmax)/2)-3, int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                    else:
                        num_immature_blueberries_cluster += 1
                        # cv2.putText(img_org, immaturity_text, (int((xmin+xmax)/2)-3, int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # img_org = cv2.addWeighted(img_org, 1, mask_overlay_all, 0.15, 0)
            # img_org = cv2.addWeighted(img_org, 1, mask_overlay, 0.3, 0)
            
            
            # Calculate the maturity ratio
            num_mature_blueberries_image += num_mature_blueberries_cluster
            num_immature_blueberries_image += num_immature_blueberries_cluster
            total_blueberries_cluster = num_mature_blueberries_cluster + num_immature_blueberries_cluster
            if total_blueberries_cluster > 0:
                maturity_ratio = num_mature_blueberries_cluster / total_blueberries_cluster
            else:
                maturity_ratio = 0.0
            print("current cluster Maturity Ratio: {:.2f}%".format(maturity_ratio * 100))

            cluster_image = crop_cluster(img_org, min_rect)
            # Scale the cluster image by 2 times
            cluster_image = cv2.resize(cluster_image, None, fx=2, fy=2)
            # Create a white rectangle with the same size as the cropped cluster image
            info_panel = np.ones_like(cluster_image) * 255
            # Concatenate the cropped cluster image and the white rectangle horizontally
            combined_image = np.hstack((cluster_image, info_panel))



            cluster_image_name = "density_map_cluster_top_FLR-12-89_" + str(cluster_num) + ".jpg"
            plant_ID = "FLR-12-89"
            cluster_ID = "topview_cluster"+ str(cluster_num)
            cluster_number = str(total_blueberries_cluster)
            cluster_maturity = str(round(maturity_ratio,2))
            cluster_compactness = str(round(cluster_mask_area/(min_rect_width* min_rect_height),2))
            
            # Define the text and starting Y-coordinate for text placement
            text_info = [
                ("Plant_ID: ", plant_ID ),
                ("Cluster_ID: ", cluster_ID),
                ("Number: ", cluster_number),
                ("Maturity: ", cluster_maturity),
                ("Compactness: ", cluster_compactness)
            ]
            y = int(cluster_image.shape[0]/2)-60
            # Iterate through the text_info and put the text on the white rectangle
            for label, text in text_info:
                cv2.putText(combined_image, label + text, (cluster_image.shape[1] + 10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y += 20


            cv2.imwrite(cluster_image_name, combined_image)
            print(cluster_image_name)
            print("--------------------------------------")
                
    cv2.imwrite('cluster_top_FLR-12-89_2.png', img_org)
    print("density_map_cluster_top_FLR-12-89.png saved.")

def crop_cluster(image, rect):
    # Get the box corners
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Order the points in the rectangle
    rect_ordered = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect_ordered[0] = box[np.argmin(s)]
    rect_ordered[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect_ordered[1] = box[np.argmin(diff)]
    rect_ordered[3] = box[np.argmax(diff)]

    # Determine the width and height
    (tl, tr, br, bl) = rect_ordered
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # Apply the perspective transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
          

# image_path = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230509/test/images/top_FLR-12-89_JPG.rf.5fa4b0013fc4f42d70236780f11c1264.jpg'

image_path ="/blue/lift-phenomics/zhengkun.li/blueberry_project/data/ASABE_dataset/blueberry_crop_single_20230509/test/images/top_FLR-12-89_JPG.rf.5fa4b0013fc4f42d70236780f11c1264.jpg"

opt = parse_opt()
model = YOLOV8(weight=opt.weight)

yolov8_boxex, yolov8_class_id, image = yolov8_detection(model, image_path)
input_boxes = torch.tensor(yolov8_boxex, device=model.device)

sam_checkpoint = "/blue/lift-phenomics/zhengkun.li/sam/segment-anything/check_point/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Create a blank mask overlay with the same shape as the input image
mask_overlay = np.zeros_like(image)

# Define a list of colors for the masks
colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]


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
mask_overlay_alpha = cv2.addWeighted(image, 1, mask_overlay, 0.3, 0)

# Save the image with the color-coded mask overlay
output_path = 'mask_overlay_top_FLR-12-89_all_mask1.jpg'
cv2.imwrite(output_path, mask_overlay_alpha)
print("Color-coded mask overlay image saved.")


blueberry_cluster(img_org = cv2.imread(image_path), boxes = yolov8_boxex, masks = masks, contour_area_filter = 4000)
