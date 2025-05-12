import os
import torch
import pydicom
import numpy as np
import albumentations as A
import torch.nn as nn
import timm
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# -------------------------
# Default Paths
# -------------------------
MODEL_PATH = "hybrid_lidc_model.pth"
DEFAULT_IMAGE_PATH = "C:/Users/Badari/OneDrive/Desktop/SDP/lung/dataset/images/images/LIDC-IDRI-0130-000001.dcm"

# -------------------------
# Hybrid Model Definition (same as training)
# -------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.efficientnet_out = self.efficientnet.num_features
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.vit_out = self.vit.num_features
        self.bilstm = nn.LSTM(input_size=self.efficientnet_out + self.vit_out, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_class = nn.Linear(512, 1)  # Binary classification
        self.fc_bbox = nn.Linear(512, 4)  # Bounding box coordinates (x, y, w, h)

    def forward(self, x):
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((eff_features, vit_features), dim=1).unsqueeze(1)
        lstm_out, _ = self.bilstm(combined_features)
        lstm_out = lstm_out[:, -1, :]
        classification = self.fc_class(lstm_out)
        bbox = self.fc_bbox(lstm_out)
        return classification, bbox

# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_dicom(dicom_path):
    # Load DICOM image
    dicom_image = pydicom.dcmread(dicom_path)
    image = dicom_image.pixel_array  # Extract pixel data
    
    # Extract projection information (if available)
    projection = None
    try:
        # Look for view position in DICOM tags
        if hasattr(dicom_image, 'ViewPosition'):
            view_position = dicom_image.ViewPosition
            if view_position == 'AP':
                projection = 'Frontal'
            elif view_position == 'PA':
                projection = 'Frontal'
            elif view_position == 'LATERAL':
                projection = 'Lateral'
            else:
                projection = view_position
        else:
            # Alternative tags that might contain projection info
            for tag in ['ViewPosition', 'PatientPosition', 'PositionerPrimaryAngle', 'ProjectionEponymousName']:
                if hasattr(dicom_image, tag):
                    projection = getattr(dicom_image, tag)
                    break
    except:
        pass
    
    # Default projection if not found
    if not projection:
        projection = 'Frontal'  # Default assumption
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize to 0-255
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    image = image.astype(np.uint8)
    
    # Apply transformations
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    transformed_image = transform(image=image)["image"]
    
    # Convert to PyTorch tensor
    tensor_image = torch.tensor(transformed_image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor_image, image, projection, dicom_image

# -------------------------
# Prediction Function
# -------------------------
def predict_nodule(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        class_output, bbox_output = model(image_tensor)
        
        # Process classification result
        prob = torch.sigmoid(class_output).item()
        prediction = "Nodule" if prob > 0.5 else "No Nodule"
        
        # Process bounding box
        bbox = bbox_output.squeeze().cpu().numpy()
        
    return prediction, prob, bbox

# -------------------------
# Visualization Function
# -------------------------
def visualize_results(original_image, prediction, probability, bbox, projection, output_path=None):
    # Convert original image to PIL for visualization
    if isinstance(original_image, np.ndarray):
        pil_image = Image.fromarray(original_image)
    else:
        pil_image = original_image
    
    # Create a drawing object
    draw = ImageDraw.Draw(pil_image)
    
    # Normalize bbox coordinates to image dimensions
    h, w = original_image.shape[:2]
    x, y, width, height = bbox
    
    # Scale to image size (only if probability > threshold)
    if probability > 0.5:
        x_min = max(0, int((x + 0.5) * w))
        y_min = max(0, int((y + 0.5) * h))
        x_max = min(w, int((x + 0.5 + width) * w))
        y_max = min(h, int((y + 0.5 + height) * h))
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    
    # Add prediction, probability and projection text
    draw.text((10, 10), f"Prediction: {prediction}", fill="red")
    draw.text((10, 30), f"Probability: {probability:.4f}", fill="red")
    draw.text((10, 50), f"Projection: {projection}", fill="red")
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(pil_image)
    plt.axis('off')
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
    
    plt.show()
    
    return pil_image

# -------------------------
# Extract DICOM Metadata
# -------------------------

# -------------------------
# Main Function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict lung nodules from DICOM images")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Path to DICOM image file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--output", help="Path to save visualization results")
    args = parser.parse_args()
    
    # Check if DICOM file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Model loaded from {args.model}")
    
    # Process image
    image_tensor, original_image, projection, dicom_image = preprocess_dicom(args.image)
    
    
    
    # Make prediction
    prediction, probability, bbox = predict_nodule(model, image_tensor, device)
    
    # Print results
    print(f"Image: {args.image}")
    print(f"Projection: {projection}")
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")
    print(f"Bounding Box (x, y, w, h): {bbox}")
    
    
    
    # Visualize results
    visualize_results(original_image, prediction, probability, bbox, projection, args.output)

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    main()