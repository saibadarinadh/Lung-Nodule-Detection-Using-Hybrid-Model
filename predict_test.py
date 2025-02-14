import torch
import os
import pydicom
import numpy as np
import albumentations as A
from timm import create_model
import argparse

# Model Path
MODEL_PATH = "lidc_model.pth"

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model("resnet18", pretrained=True, num_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Define Transformations
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# Function to Predict on a Single Image
def predict(image_path):
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return

    # Load DICOM Image
    dicom_image = pydicom.dcmread(image_path)
    image = dicom_image.pixel_array  # Extract pixel data

    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Normalize to 0-255
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    image = image.astype(np.uint8)

    # Apply transformations
    image = transform(image=image)["image"]

    # Convert to PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = torch.sigmoid(model(image)).item()

    # Print Prediction
    if output > 0.9:
        print(f"Prediction: **Lung Nodule Detected (Confidence: {output:.2f})**")
    else:
        print(f"Prediction: **No Lung Nodule (Confidence: {1 - output:.2f})**")


# Get Input Image from User
if __name__ == "__main__":
    path="./dataset/images/images/LIDC-IDRI-0262-000001.dcm"
    predict(path)
