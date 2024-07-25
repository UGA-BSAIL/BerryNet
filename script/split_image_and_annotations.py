import cv2
import os

def get_bbox_area(x_center, y_center, width, height, x_start, y_start, x_end, y_end):
    # calculate the coordinates of the bounding box in the segmented image
    x1 = max(x_start, x_center - width / 2)
    y1 = max(y_start, y_center - height / 2)
    x2 = min(x_end, x_center + width / 2)
    y2 = min(y_end, y_center + height / 2)

    # calculate the area of the bounding box
    return max(0, x2 - x1) * max(0, y2 - y1)

def split_image_and_annotations(image_path, txt_path, output_folder, grid_size=(3, 3), area_threshold=0.5):
    # read the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # read the annotation file
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # calculate the size of each tile
    tile_width, tile_height = w // grid_size[1], h // grid_size[0]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # calculate the coordinates of the tile
            x_start, y_start = j * tile_width, i * tile_height
            x_end, y_end = (j + 1) * tile_width, (i + 1) * tile_height

            # crop the tile
            tile = image[y_start:y_end, x_start:x_end]

            # process each annotation
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # convert to absolute coordinates
                x_center, y_center, width, height = x_center*w, y_center*h, width*w, height*h

                #  check if the annotation is in the current tile
                if x_start <= x_center < x_end and y_start <= y_center < y_end:
                    # adjust the bbox to the segmented image
                    new_x_center, new_y_center, new_width, new_height = adjust_bbox_in_tile(
                        x_center, y_center, width, height, x_start, y_start, x_end, y_end)

                    # calculate the adjusted bbox area
                    adjusted_area = new_width * new_height
                    original_area = width * height

                    # only add the annotation if the bbox occupies enough area in the segmented image
                    if adjusted_area / original_area >= area_threshold:
                        # convert the coordinates and sizes to normalized values relative to the segmented image
                        new_line = f"{int(class_id)} {(new_x_center - x_start) / tile_width} {(new_y_center - y_start) / tile_height} {new_width / tile_width} {new_height / tile_height}\n"
                        new_lines.append(new_line)




            # 保存小图像和标注
            tile_name = f"{os.path.splitext(image_file)[0]}_{i}_{j}.jpg"
            tile_path = os.path.join(output_folder, tile_name)
            cv2.imwrite(tile_path, tile)

            tile_txt_name = f"{os.path.splitext(image_file)[0]}_{i}_{j}.txt"
            tile_txt_path = os.path.join(output_folder, tile_txt_name)
            with open(tile_txt_path, 'w') as f:
                f.writelines(new_lines)

def adjust_bbox_in_tile(x_center, y_center, width, height, x_start, y_start, x_end, y_end):
    # 调整边界框坐标以适应分割图像
    x1 = max(x_center - width / 2, x_start)
    y1 = max(y_center - height / 2, y_start)
    x2 = min(x_center + width / 2, x_end)
    y2 = min(y_center + height / 2, y_end)

    # 新的中心点和尺寸
    new_x_center = (x1 + x2) / 2
    new_y_center = (y1 + y2) / 2
    new_width = x2 - x1
    new_height = y2 - y1

    return new_x_center, new_y_center, new_width, new_height

# 设置文件夹路径
image_folder = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/dataset/segmentation/ROBOset_20230509_test_N39/test/images'
annotation_folder = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/dataset/segmentation/ROBOset_20230509_test_N39/test/labels'
output_folder = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/dataset/segmentation/ROBOset_20230509_test_N39/test/split'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个图像和标签
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))
        
        if os.path.exists(txt_path):
            split_image_and_annotations(img_path, txt_path, output_folder)
