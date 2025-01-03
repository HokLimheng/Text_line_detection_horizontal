'''
This code will detect lines in the image
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_width = 1400
image_height = 100

# Define the input image path (directory) and output directory
input_image_path = 'input_image_path'
# input_image_path = 'test_img.png'
output_dir = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
image = cv2.imread(input_image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Define a horizontal kernel to detect horizontal lines of textn
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))

# Apply morphological operations to connect components horizontally
dilated = cv2.dilate(binary, kernel, iterations=3)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and save each detected vertical line as an image
# for i, contour in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Define thresholds
min_width = 45
min_height = 5
max_height_for_split = 25  # Maximum height before splitting the contour

# Process contours
for contour in contours:
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
            
            # Only draw and process if the split height meets minimum requirements
            if split_h > min_height and w > min_width:
                cv2.rectangle(image, (x, split_y), (x + w, split_y + split_h), (255, 0, 0), 2)
    elif w > min_width and h > min_height:
        # Draw the bounding box for contours within normal height
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        

# Display the image with bounding boxes
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Bounding Boxes around Text Lines - {input_image_path}")
plt.axis("off")
plt.show()

# Optional: Pause for user input to proceed to the next image
# input("Press Enter to continue to the next image...")
