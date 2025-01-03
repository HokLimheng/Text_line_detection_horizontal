''''
This code will cut images, and save all of them in a more organized way. Example: if the image has 3 lines, 
it will save all of them in a image1 with "line_1.png", "line_2.png", etc.

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
input_image_folder = "input_image_folder"  # Folder containing images to process
output_base_dir = "output"  # Base output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Parameters for line detection
min_width = 45
min_height = 5
max_height_for_split = 25  # Maximum height before splitting the contour

# Iterate through all images in the input folder
for image_file in os.listdir(input_image_folder):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        input_image_path = os.path.join(input_image_folder, image_file)
        
        # Read the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error loading {image_file}. Skipping.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Define a vertical kernel to detect vertically aligned lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))

        # Apply morphological operations to highlight lines of text
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # Find contours for the vertical lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by their vertical position (y coordinate)
        sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

        # Create a subdirectory for this image's lines
        image_name = os.path.splitext(image_file)[0]  # Get the name without extension
        output_dir = os.path.join(output_base_dir, image_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process contours and save lines
        line_count = 1
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour height suggests it might contain multiple lines
            if h > max_height_for_split:
                # Calculate the number of vertical splits needed
                num_splits = int(h / max_height_for_split) + 1
                split_height = h // num_splits

                # Split the contour into smaller segments vertically
                for j in range(num_splits):
                    split_y = y + j * split_height
                    split_h = split_height if j < num_splits - 1 else h - j * split_height
                    
                    # Only save the split contour if it meets minimum requirements
                    if split_h > min_height and w > min_width:
                        line_roi = image[split_y:split_y + split_h, x:x + w]
                        output_path = os.path.join(output_dir, f"line_{line_count}.png")
                        cv2.imwrite(output_path, line_roi)
                        line_count += 1
            elif w > min_width and h > min_height:
                # Save the bounding box for contours within normal height
                line_roi = image[y:y + h, x:x + w]
                output_path = os.path.join(output_dir, f"line_{line_count}.png")
                cv2.imwrite(output_path, line_roi)
                line_count += 1

        print(f"Processed {image_file}: {line_count - 1} lines saved in {output_dir}")

# Display a success message
print("Processing complete. All lines have been saved.")
