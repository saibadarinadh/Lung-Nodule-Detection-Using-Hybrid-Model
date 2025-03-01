import os
import torch
import pydicom
import numpy as np
import albumentations as A
import timm
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# Create Flask app
app = Flask(__name__, static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------
# Hybrid Model Definition
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
model = None

def load_model():
    """Load the model only when needed to save memory"""
    global model
    if model is None:
        model = HybridModel().to(device)
        model.load_state_dict(torch.load("hybrid_lidc_model.pth", map_location=device))
        model.eval()
        print("Model loaded successfully")
    return model

# -------------------------
# Define Transformations
# -------------------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# -------------------------
# Helper Functions
# -------------------------
def process_dicom_image(filepath):
    """Process a DICOM file and return the normalized pixel array and display image"""
    dicom_image = pydicom.dcmread(filepath)
    pixel_array = dicom_image.pixel_array
    
    # Convert to 3 channels if grayscale
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    
    # Normalize the image for display
    display_image = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
    display_image = display_image.astype(np.uint8)
    
    return pixel_array, display_image

def get_base64_image(image_array):
    """Convert image array to base64 encoded string"""
    pil_img = Image.fromarray(image_array[:,:,0] if image_array.shape[2] == 3 else image_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# -------------------------
# API Routes
# -------------------------
@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/preview', methods=['POST'])
def preview_image():
    """Generate a preview of the DICOM image without analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.dcm'):
        return jsonify({'error': 'File must be a DICOM (.dcm) file'}), 400
    
    if file:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the DICOM file
            _, display_image = process_dicom_image(filepath)
            
            # Create a base64 representation of the image
            img_str = get_base64_image(display_image)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            # Return the preview image
            return jsonify({
                'image': img_str
            })
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Failed to process file'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Process and analyze the uploaded DICOM image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.dcm'):
        return jsonify({'error': 'File must be a DICOM (.dcm) file'}), 400
    
    if file:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process the DICOM file
            pixel_array, display_image = process_dicom_image(filepath)
            
            # Create a base64 representation of the image
            img_str = get_base64_image(display_image)
            
            # Apply transformations for the model
            transformed_image = transform(image=display_image)["image"]
            tensor_image = torch.tensor(transformed_image).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Load model and make prediction
            model = load_model()
            with torch.no_grad():
                output = model(tensor_image).squeeze()
                probability = torch.sigmoid(output).item()
            
            # Clean up the temporary file
            os.remove(filepath)
            
            # Return prediction results
            return jsonify({
                'probability': probability,
                'prediction': "Nodule Detected" if probability > 0.5 else "No Nodules",
                'image': img_str
            })
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Failed to process file'}), 500

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)