import argparse
import cv2
import insightface
import os

# Load the InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # Use CPU, set ctx_id=0 for GPU


# Function to extract faces and include hair and neck (with padding)
def extract_faces(image_path: str, output_dir: str, padding: float = 0.3):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    # Get image dimensions
    img_height, img_width = img.shape[:2]

    # Detect faces in the image
    faces = model.get(img)

    # If no faces are detected, return
    if not faces:
        print(f"No faces detected in image: {image_path}")
        return

    # Find the largest face by comparing bounding box areas
    face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))

    # Get the face bounding box and apply padding
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    # Calculate padding around the face
    width = x2 - x1
    height = y2 - y1
    pad_x = int(padding * width)
    pad_y = int(padding * height)

    # Apply padding and ensure the coordinates don't exceed the image dimensions
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_width, x2 + pad_x)
    y2 = min(img_height, y2 + pad_y)

    # Extract the region with the face plus padding (hair/neck)
    face_img = img[y1:y2, x1:x2]

    # Save the extracted face to the output directory
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face.jpg")
    cv2.imwrite(output_path, face_img)
    print(f"Saved face to {output_path}")


# Main function to process multiple images
def process_images(input_dir, output_dir, padding=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{args.output}'")

    # Loop over each image file in the input directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            extract_faces(file_path, output_dir, padding)


# Initialize the parser
parser = argparse.ArgumentParser(description="Example of reading command-line arguments")

# Add arguments
parser.add_argument("input", help="Input directory path")
parser.add_argument("output", help="Output directory path")
parser.add_argument("--padding", type=float, default=0.3, help="Padding for face extraction (default: 0.3)")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
print(f"Input directory: {args.input}")
print(f"Output directory: {args.output}")
print(f"Padding: {args.padding}")

if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input directory '{args.input}' does not exist.")

# Run the face extraction on all images in the input directory with padding for hair/neck
process_images(args.input, args.output, args.padding)  # You can adjust the padding here
