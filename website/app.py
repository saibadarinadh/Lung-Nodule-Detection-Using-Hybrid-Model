import os
import torch
import pydicom
import numpy as np
import albumentations as A
import timm
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory, render_template, session
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import logging
from werkzeug.exceptions import BadRequest, InternalServerError
from shutil import copyfile
import matplotlib
from imagechange import update_figure, set_current_image
matplotlib.use('Agg')  # Required for Flask backend
import cv2
from manual_analysis import manual_analysis_bp
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add after app initialization
MODEL_PATH = 'C:/Users/Badari/OneDrive/Desktop/Lung-Nodule-Detection-Using-Hybrid-Model/website/saved_dicom_model.pkl'
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file {MODEL_PATH} not found!")
    print(f"WARNING: Model file {MODEL_PATH} not found!")

# -------------------------
# Updated Hybrid Model Definition
# -------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.efficientnet_out = self.efficientnet.num_features
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.vit_out = self.vit.num_features
        self.bilstm = nn.LSTM(input_size=self.efficientnet_out + self.vit_out, 
                             hidden_size=256, num_layers=2, 
                             batch_first=True, bidirectional=True)
        self.fc_class = nn.Linear(512, 1)  # Binary classification
        self.fc_bbox = nn.Linear(512, 4)   # Bounding box coordinates (x, y, w, h)

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
# Load Trained Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    """Load the model only when needed to save memory"""
    global model
    if model is None:
        model = HybridModel().to(device)
        model.load_state_dict(torch.load("C:/Users/Badari/OneDrive/Desktop/Lung-Nodule-Detection-Using-Hybrid-Model/website/hybrid_lidc_model.pth", map_location=device))
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
# Updated Helper Functions
# -------------------------
def process_dicom_image(filepath):
    """Process a DICOM file and return the normalized pixel array and display image"""
    dicom_image = pydicom.dcmread(filepath)
    pixel_array = dicom_image.pixel_array
    
    # Get projection information
    projection = 'Frontal'  # Default
    if hasattr(dicom_image, 'ViewPosition'):
        view_position = dicom_image.ViewPosition
        if view_position in ['AP', 'PA']:
            projection = 'Frontal'
        elif view_position == 'LATERAL':
            projection = 'Lateral'
        else:
            projection = view_position
    
    # Convert to 3 channels if grayscale
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    
    # Normalize the image for display
    display_image = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
    display_image = display_image.astype(np.uint8)
    
    return pixel_array, display_image, projection

def draw_bbox_on_image(image, bbox, probability):
    """Draw bounding box and predictions on the image"""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    h, w = image.shape[:2]
    x, y, width, height = bbox
    
    if probability > 0.5:
        try:
            # Calculate coordinates with validation
            x_min = max(0, int((x + 0.5) * w))
            y_min = max(0, int((y + 0.5) * h))
            x_max = min(w, int((x + 0.5 + abs(width)) * w))
            y_max = min(h, int((y + 0.5 + abs(height)) * h))
            
            # Ensure valid box coordinates
            if x_max > x_min and y_max > y_min:
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            else:
                logger.warning("Invalid bounding box coordinates detected")
        except Exception as e:
            logger.error(f"Error drawing bounding box: {str(e)}")
    
    # Add prediction and probability text with safe positioning
    try:
        # Draw text with padding and background
        text_padding = 10
        prediction_text = f"Prediction: {'Nodule' if probability > 0.5 else 'No Nodule'}"
        probability_text = f"Probability: {probability:.4f}"
        
        # Add background for better text visibility
        text_bg_color = (0, 0, 0, 127)  # Semi-transparent black
        text_color = "white"
        
        # Get text sizes
        pred_bbox = draw.textbbox((0, 0), prediction_text)
        prob_bbox = draw.textbbox((0, 0), probability_text)
        
        # Draw text backgrounds
        draw.rectangle([text_padding-5, text_padding-5, 
                       text_padding + pred_bbox[2]+5, text_padding + pred_bbox[3]+5], 
                      fill=text_bg_color)
        draw.rectangle([text_padding-5, text_padding + pred_bbox[3] + 10-5,
                       text_padding + prob_bbox[2]+5, text_padding + pred_bbox[3] + prob_bbox[3] + 10+5],
                      fill=text_bg_color)
        
        # Draw text
        draw.text((text_padding, text_padding), prediction_text, fill=text_color)
        draw.text((text_padding, text_padding + pred_bbox[3] + 10), probability_text, fill=text_color)
        
    except Exception as e:
        logger.error(f"Error adding text to image: {str(e)}")
    
    return np.array(pil_image)

def get_base64_image(image_array):
    """Convert image array to base64 encoded string"""
    pil_img = Image.fromarray(image_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def load_and_enhance_xray(filepath):
    """Load and enhance a DICOM X-ray image for manual analysis"""
    try:
        # Load and process the DICOM file
        _, display_image, _ = process_dicom_image(filepath)
        
        # Convert the processed image to base64
        img_str = get_base64_image(display_image)
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error in load_and_enhance_xray: {str(e)}")
        return None

def apply_basic_adjustments(image, brightness=0, contrast=1.0, gamma=1.0):
    """Apply basic image adjustments"""
    image = image.astype(float)
    # Apply contrast
    image = image * contrast
    # Apply brightness
    image = image + (brightness * 255)
    # Apply gamma correction
    if gamma != 1.0:
        image = np.power(image/255, gamma) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def apply_denoising(image, median_blur=0, gaussian_blur=0, bilateral=0):
    """Apply denoising filters"""
    if median_blur > 0:
        kernel_size = 2 * int(median_blur) + 1
        image = cv2.medianBlur(image, kernel_size)
    if gaussian_blur > 0:
        kernel_size = 2 * int(gaussian_blur) + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if bilateral > 0:
        image = cv2.bilateralFilter(image, bilateral, 75, 75)
    return image

def apply_enhancement(image, clahe_clip=0, sharpen=0):
    """Apply image enhancements"""
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        if len(image.shape) == 3:
            # For RGB images, apply CLAHE to luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            image = clahe.apply(image)
    
    if sharpen > 0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpen
        image = cv2.filter2D(image, -1, kernel)
    return image

def apply_edge_detection(image, canny=False, sobel=False, laplacian=False, threshold=0):
    """Apply edge detection"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    if canny:
        image = cv2.Canny(gray, 100, 200)
    if sobel:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        image = cv2.magnitude(sobelx, sobely)
    if laplacian:
        image = cv2.Laplacian(gray, cv2.CV_64F)
    if threshold > 0:
        _, image = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Convert back to 3 channels if input was RGB
    if len(image.shape) == 2 and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def apply_morphology(image, operation, kernel_size):
    """Apply morphological operations"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def check_image_relevance(image_path, model_path=MODEL_PATH, similarity_threshold=0.95):
    """Check if the image is related to the dataset"""
    try:
        # Load the VGG16 model for feature extraction
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Extract features from test image
        img = process_dicom_image(image_path)[1]  # Get the processed display image
        img = Image.fromarray(img).resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        test_features = vgg_model.predict(img_array, verbose=0).flatten()
        
        # Load the saved feature model
        try:
            with open(model_path, 'rb') as file:
                dataset_features = pickle.load(file)
        except FileNotFoundError:
            logger.error(f"Model file {model_path} not found")
            return False, "Model file not found. Please contact support."
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False, "Error loading model. Please try again."
            
        # Calculate similarity
        similarities = cosine_similarity([test_features], dataset_features)
        max_similarity = np.max(similarities)
        
        logger.info(f"Image similarity score: {max_similarity}")
        
        if max_similarity < similarity_threshold:
            return False, f"Image appears to be unrelated to lung CT scans (similarity: {max_similarity:.2f}). Please upload a valid lung X-Ray image."
        
        return True, "Image verified as lung CT scan"
        
    except Exception as e:
        logger.error(f"Error checking image relevance: {str(e)}")
        return False, "Error processing image. Please ensure you're uploading a valid DICOM file."

# -------------------------
# Updated API Routes
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
@app.route('/api/preview', methods=['POST'])
def preview():
    """Generate preview for uploaded DICOM file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not file.filename.endswith('.dcm'):
            return jsonify({'error': 'File must be a DICOM (.dcm) file'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            # Process DICOM image
            _, display_image, _ = process_dicom_image(filepath)
            
            # Convert to base64
            img_str = get_base64_image(display_image)
            
            # Store current image for manual analysis
            global current_image, current_figure
            current_image = display_image
            current_figure = img_str
            
            return jsonify({
                'success': True,
                'image': img_str
            })
            
        except Exception as e:
            logger.error(f"Error processing DICOM for preview: {str(e)}")
            return jsonify({'error': 'Failed to process DICOM file'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        return jsonify({'error': 'Failed to generate preview'}), 500
    
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Process and analyze the uploaded DICOM image"""
    try:
        if 'file' not in request.files:
            raise BadRequest('No file uploaded')
        
        file = request.files['file']
        
        if file.filename == '':
            raise BadRequest('Empty filename')
        
        if not file.filename.endswith('.dcm'):
            raise BadRequest('Invalid file format - must be DICOM (.dcm)')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # First check if image is related to dataset
            is_related, message = check_image_relevance(filepath)
            
            if not is_related:
                # If not related, return early with notification
                return jsonify({
                    'error': 'Invalid Image Type',
                    'details': message,
                    'code': 'INVALID_IMAGE'
                }), 400
            
            # If related, proceed with normal analysis
            try:
                pixel_array, display_image, projection = process_dicom_image(filepath)
            except Exception as e:
                logger.error(f"Error processing DICOM file: {str(e)}")
                raise InternalServerError('Failed to process DICOM file')

            # Model prediction
            try:
                transformed_image = transform(image=display_image)["image"]
                tensor_image = torch.tensor(transformed_image).permute(2, 0, 1).unsqueeze(0).to(device)
                
                model = load_model()
                with torch.no_grad():
                    class_output, bbox_output = model(tensor_image)
                    probability = torch.sigmoid(class_output).squeeze().item()
                    bbox = bbox_output.squeeze().cpu().numpy()
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
                raise InternalServerError('Failed to analyze image')

            # Generate result image
            try:
                annotated_image = draw_bbox_on_image(display_image, bbox, probability)
                if annotated_image is None:
                    raise ValueError("Failed to generate annotated image")
                img_str = get_base64_image(annotated_image)
            except Exception as e:
                logger.error(f"Error generating result image: {str(e)}")
                img_str = get_base64_image(display_image)
                logger.info("Falling back to original image")

            return jsonify({
                'probability': probability,
                'prediction': "Nodule Detected" if probability > 0.5 else "No Nodules",
                'bbox': bbox.tolist(),
                'projection': projection,
                'image': img_str
            })

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise InternalServerError('An unexpected error occurred')
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {str(e)}")

    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except InternalServerError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/manual-analysis')
def manual_analysis_page():
    """Serve the manual analysis page"""
    try:
        return render_template('manual_analysis.html')
    except Exception as e:
        logger.error(f"Error rendering manual analysis template: {str(e)}")
        return jsonify({'error': 'Failed to load manual analysis page'}), 500

# Add this global variable to store the current figure
current_figure = None
current_image = None  # Add this new global variable

@app.route('/api/get-analysis-figure')
def get_analysis_figure():
    """Get the current analysis figure"""
    global current_figure
    if current_figure is None:
        return jsonify({'error': 'No figure available'}), 404
    
    # Make sure the image data is properly formatted
    if not current_figure.startswith('data:image'):
        current_figure = f'data:image/png;base64,{current_figure}'
        
    return jsonify({
        'figure': current_figure
    })

@app.route('/api/manual-analysis', methods=['POST'])
def start_manual_analysis():
    """Process the image for manual analysis"""
    global current_figure, current_image  # Add current_image to globals
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.dcm'):
            return jsonify({'error': 'Invalid file format'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process the image
            _, display_image, _ = process_dicom_image(filepath)
            
            # Store the current image globally
            current_image = display_image
            
            # Convert to base64
            current_figure = get_base64_image(display_image)
            
            return jsonify({
                'success': True,
                'figure': current_figure
            })
            
        except Exception as e:
            logger.error(f"Error in manual analysis: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/update-analysis', methods=['POST'])
def update_analysis():
    """Update the analysis figure with new parameters"""
    global current_image, current_figure
    
    try:
        if current_image is None:
            return jsonify({'error': 'No image loaded'}), 400

        # Get filter parameters from request
        data = request.json
        
        # Make a copy of the original image
        processed_image = current_image.copy()
        
        # Apply basic adjustments
        processed_image = apply_basic_adjustments(
            processed_image,
            brightness=float(data.get('brightness', 0)),
            contrast=float(data.get('contrast', 1.0)),
            gamma=float(data.get('gamma', 1.0))
        )
        
        # Apply denoising if needed
        if data.get('medianBlur', 0) > 1:
            kernel_size = int(data.get('medianBlur'))
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed_image = cv2.medianBlur(processed_image, kernel_size)
            
        if data.get('gaussianBlur', 0) > 0:
            kernel_size = int(data.get('gaussianBlur')) * 2 + 1
            processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), 0)
            
        if data.get('bilateralD', 0) > 0:
            processed_image = cv2.bilateralFilter(
                processed_image,
                data.get('bilateralD'),
                data.get('bilateralSigmaColor', 75),
                data.get('bilateralSigmaSpace', 75)
            )
            
        # Apply enhancement filters
        if data.get('clahe', 0) > 0:
            clahe = cv2.createCLAHE(clipLimit=float(data.get('clahe')), tileGridSize=(8,8))
            if len(processed_image.shape) == 3:
                lab = cv2.cvtColor(processed_image, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                processed_image = clahe.apply(processed_image)
                
        # Apply edge detection
        if data.get('cannyEnabled'):
            processed_image = cv2.Canny(
                processed_image,
                data.get('cannyThreshold1', 100),
                data.get('cannyThreshold2', 200)
            )
            
        # Convert to base64 and return
        processed_figure = get_base64_image(processed_image)
        
        return jsonify({
            'success': True,
            'figure': processed_figure
        })
        
    except Exception as e:
        logger.error(f"Error updating analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

def apply_image_adjustments(image, method, contrast, brightness, gamma, enhancement=1.0, 
                          sharpness=1.0, denoise=0.0, nodule_sensitivity=1.0, threshold=0.5):
    """Apply image adjustments based on parameters"""
    try:
        # Convert to float for calculations
        image = image.astype(float)
        
        # Apply contrast
        image = image * contrast
        
        # Apply brightness
        image = image + (brightness * 255)
        
        # Apply gamma
        if (gamma != 1.0):
            image = np.power(image/255, gamma) * 255
            
        # Apply enhancement
        if enhancement != 1.0:
            clahe = cv2.createCLAHE(clipLimit=2.0 * enhancement, tileGridSize=(8,8))
            image = clahe.apply(image.astype(np.uint8)).astype(float)
            
        # Apply sharpening
        if sharpness != 1.0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpness
            image = cv2.filter2D(image.astype(np.uint8), -1, kernel).astype(float)
            
        # Apply denoising
        if denoise > 0:
            image = cv2.fastNlMeansDenoising(image.astype(np.uint8), 
                                           None, 
                                           denoise * 10,
                                           7,
                                           21).astype(float)
            
        # Apply method-specific enhancements with sensitivity
        if method == "Nodule Detection":
            # Apply Gaussian blur with sensitivity
            kernel_size = int(3 + (2 * nodule_sensitivity))
            if kernel_size % 2 == 0: kernel_size += 1  # Ensure odd kernel size
            image = cv2.GaussianBlur(image.astype(np.uint8), 
                                   (kernel_size, kernel_size), 
                                   nodule_sensitivity).astype(float)
            
            # Apply threshold
            if threshold < 1.0:
                _, image = cv2.threshold(image.astype(np.uint8), 
                                      int(255 * threshold), 
                                      255, 
                                      cv2.THRESH_BINARY)
                image = image.astype(float)
            
        # Clip values to valid range
        image = np.clip(image, 0, 255)
        
        return image.astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error in apply_image_adjustments: {str(e)}")
        return image.astype(np.uint8)

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)

app.register_blueprint(manual_analysis_bp)

@manual_analysis_bp.route('/api/manual-analysis-process', methods=['POST'])
def manual_analysis_process():
    dicom_path = session.get('last_dicom_path')
    if not dicom_path or not os.path.exists(dicom_path):
        return jsonify({'error': 'No DICOM image found. Please upload on the home page.'}), 400
    img = load_dicom_image(dicom_path)
    # ... apply filters ...
    Image.fromarray(img).save('static/original_image.png')
    # ... save processed image as 'static/processed_image.png' ...
    return jsonify({'processed_image_url': '/static/processed_image.png'})

@app.route('/api/manual-analysis-process', methods=['POST'])
def process_manual_analysis():
    """Process image with selected filters"""
    global current_image, current_figure
    
    try:
        if current_image is None:
            return jsonify({'error': 'No image loaded'}), 400

        # Get filter parameters from request
        data = request.json
        
        # Make a copy of the original image
        processed_image = current_image.copy()
        
        # Apply filters based on parameters
        if data.get('brightness') or data.get('contrast') or data.get('gamma'):
            processed_image = apply_basic_adjustments(
                processed_image,
                brightness=float(data.get('brightness', 0)),
                contrast=float(data.get('contrast', 1.0)),
                gamma=float(data.get('gamma', 1.0))
            )
            
        if data.get('median') > 1:
            kernel_size = int(data.get('median'))
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed_image = cv2.medianBlur(processed_image, kernel_size)
            
        if data.get('gaussian') > 0:
            kernel_size = int(data.get('gaussian')) * 2 + 1
            processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), 0)
            
        if data.get('clahe') > 0:
            clahe = cv2.createCLAHE(clipLimit=float(data.get('clahe')), tileGridSize=(8,8))
            if len(processed_image.shape) == 3:
                # For RGB images, apply CLAHE to luminance channel
                lab = cv2.cvtColor(processed_image, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                processed_image = clahe.apply(processed_image)

        # Convert to base64 and return
        processed_image_base64 = get_base64_image(processed_image)
        
        return jsonify({
            'success': True,
            'processed_image_url': processed_image_base64
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500