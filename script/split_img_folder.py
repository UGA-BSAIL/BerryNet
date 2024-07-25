import cv2
import os

# Define the path to the folder containing the images
folder_path = '/blue/lift-phenomics/zhengkun.li/blueberry_project/data/flower/flower_dataset_1/org'
patch_num = 2   # 2*2

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
else:
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can modify this check based on your image file extensions)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG')):
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)

            # Read the image
            image = cv2.imread(image_path)

            # Check if the image was successfully loaded
            if image is None:
                print(f"Error: Could not read the image '{image_path}'.")
            else:
                # Get the dimensions of the image
                height, width, _ = image.shape

                # Define the size of each small patch
                patch_size_x = width // patch_num
                patch_size_y = height // patch_num

                # Create a directory to save the patches if it doesn't exist
                patches_folder = os.path.join(folder_path, 'patches')
                if not os.path.exists(patches_folder):
                    os.makedirs(patches_folder)

                # Loop through the image and extract and save each small patch
                for y in range(patch_num):
                    for x in range(patch_num):
                        # Calculate the coordinates for the top-left and bottom-right corners of the patch
                        top_left_x = x * patch_size_x
                        top_left_y = y * patch_size_y
                        bottom_right_x = (x + 1) * patch_size_x
                        bottom_right_y = (y + 1) * patch_size_y

                        # Extract the patch
                        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                        # Save the patch with a unique filename
                        patch_filename = os.path.join(patches_folder, f'patch_{y}{x}_{filename}')
                        print(patch_filename)
                        cv2.imwrite(patch_filename, patch)
                        print(f'Saved: {patch_filename}')

    print("All patches saved successfully.")
