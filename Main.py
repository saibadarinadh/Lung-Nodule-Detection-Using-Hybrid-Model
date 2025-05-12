import os
import torch
import pydicom
import numpy as np
import albumentations as A
import timm
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# -------------------------
# Load the Hybrid Model
# -------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        
        # Feature extractor - EfficientNet
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)  
        self.efficientnet_out = self.efficientnet.num_features  

        # Transformer - Vision Transformer (ViT)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.vit_out = self.vit.num_features  

        # BiLSTM
        self.bilstm = nn.LSTM(input_size=self.efficientnet_out + self.vit_out, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        eff_features = self.efficientnet(x)  
        vit_features = self.vit(x)
        combined_features = torch.cat((eff_features, vit_features), dim=1).unsqueeze(1)  
        lstm_out, _ = self.bilstm(combined_features)
        lstm_out = lstm_out[:, -1, :]  
        output = self.fc(lstm_out)
        return output

# -------------------------
# Load Trained Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel().to(device)
model.load_state_dict(torch.load("./website/hybrid_lidc_model.pth", map_location=device))
model.eval()

# -------------------------
# Define Transformations
# -------------------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# -------------------------
# Function to Predict Image
# -------------------------
def predict_image(image_path):
    dicom_image = pydicom.dcmread(image_path)
    image = dicom_image.pixel_array  

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    image = image.astype(np.uint8)

    image = transform(image=image)["image"]

    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(image).squeeze()
        probability = torch.sigmoid(output).item()

    prediction = "Nodule Detected ✅" if probability > 0.5 else "No Nodules ❌"
    return prediction, probability

# -------------------------
# GUI Interface
# -------------------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
    if file_path:
        prediction, confidence = predict_image(file_path)
        result_label.config(text=f"🔍 Prediction: {prediction}\nConfidence: {confidence:.4f}")
        
        # Display DICOM image
        dicom_image = pydicom.dcmread(file_path)
        image = dicom_image.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = image.astype(np.uint8)
        img = Image.fromarray(image).convert("L")
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create Tkinter Window
root = tk.Tk()
root.title("Lung Nodule Detection")
root.geometry("400x500")
root.configure(bg="white")

# Title Label
title_label = Label(root, text="Lung Nodule Detection", font=("Arial", 16, "bold"), bg="white")
title_label.pack(pady=10)

# Upload Button
upload_button = Button(root, text="Upload DICOM Image", command=upload_and_predict, font=("Arial", 12), bg="blue", fg="white")
upload_button.pack(pady=10)

# Image Label
image_label = Label(root, bg="white")
image_label.pack(pady=10)

# Result Label
result_label = Label(root, text="Prediction will appear here.", font=("Arial", 12), bg="white")
result_label.pack(pady=10)

# Run GUI
root.mainloop()
