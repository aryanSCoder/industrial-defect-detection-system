import cv2
import os
import glob

# Path containing your dataset images
IMAGE_FOLDER = r"dataset\NEU-DET\IMAGES"

# Output video path
OUTPUT_VIDEO = r"test_defect_video.mp4"

# Get all JPG images
images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

# Sort images
images.sort()

# Select the first 50 images
images = images[:50]

if not images:
    print("No images found!")
    exit()

# Read the first image to get dimensions
first_image = cv2.imread(images[0])

height, width, _ = first_image.shape

# Video settings
fps = 5

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

print("Creating video...")

# Add every image to the video
for index, image_path in enumerate(images):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    # Ensure every image has the same size
    image = cv2.resize(image, (width, height))

    # Add image to video
    video.write(image)

    print(f"Added image {index + 1}/{len(images)}")

video.release()

print("\nVideo created successfully!")
print(f"Saved as: {OUTPUT_VIDEO}")