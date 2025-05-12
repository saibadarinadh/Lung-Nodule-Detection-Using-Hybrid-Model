import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
import os
from scipy import ndimage
from skimage import exposure, filters, feature, restoration, util
from io import BytesIO
import base64
import pydicom
import logging
from PIL import Image
from imageprocessing import (
    apply_basic_adjustments, apply_denoising, apply_enhancement,
    apply_histogram_equalization, apply_adaptive_threshold,
    apply_sobel, apply_laplacian, apply_blob_detection
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the current image
current_image = None
current_enhanced = None
current_figure = None

def load_and_enhance_xray(filepath):
    """Load and enhance the X-ray image"""
    global current_image, current_enhanced
    
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(filepath)
        current_image = dicom.pixel_array
        
        # Normalize image
        current_image = (current_image - np.min(current_image)) / (np.max(current_image) - np.min(current_image))
        
        # Apply initial enhancement
        current_enhanced = apply_clahe(current_image)
        
        return create_figure(current_enhanced)
    except Exception as e:
        logger.error(f"Error in load_and_enhance_xray: {str(e)}")
        raise

def update_analysis(params):
    """Update the analysis with the provided parameters, including new filters"""
    global current_image, current_enhanced, current_figure
    if current_image is None:
        return None
    try:
        # Extract parameters
        contrast = float(params.get('contrast', 1.0))
        brightness = float(params.get('brightness', 0.0))
        gamma = float(params.get('gamma', 1.0))
        clahe = float(params.get('clahe', 0.0))
        unsharp = float(params.get('unsharp', 0.0))
        canny = params.get('cannyEnabled', False)
        canny1 = int(params.get('cannyThreshold1', 100))
        canny2 = int(params.get('cannyThreshold2', 200))
        histeq = params.get('histeq', False)
        adaptive_thresh = params.get('adaptiveThresh', False)
        sobel = params.get('sobel', False)
        laplacian = params.get('laplacian', False)
        blob = params.get('blob', False)

        # Start with the original image
        processed = current_image.copy()
        # Clamp and apply basic adjustments
        processed = apply_basic_adjustments(processed, brightness, contrast, gamma)
        # CLAHE and unsharp
        processed = apply_enhancement(processed, clahe, unsharp)
        # Histogram Equalization
        if histeq:
            processed = apply_histogram_equalization(processed)
        # Adaptive Threshold
        if adaptive_thresh:
            processed = apply_adaptive_threshold(processed)
        # Sobel
        if sobel:
            processed = apply_sobel(processed)
        # Laplacian
        if laplacian:
            processed = apply_laplacian(processed)
        # Blob Detection
        if blob:
            processed = apply_blob_detection(processed)
        # Canny
        if canny:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            processed = cv2.Canny(processed, canny1, canny2)
        # Normalize if needed
        if processed.max() > 255 or processed.min() < 0:
            processed = np.clip(processed, 0, 255)
        # Create and return the figure
        current_figure = create_figure(processed / 255.0 if processed.max() > 1 else processed)
        return current_figure
    except Exception as e:
        logger.error(f"Error in update_analysis: {str(e)}")
        raise

def apply_clahe(image, clip_limit=2.0, grid_size=8):
    """Apply CLAHE enhancement"""
    image_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(image_uint8)
    return enhanced / 255.0

def enhance_nodules(image, sensitivity=1.0):
    """Enhance potential nodule regions using multi-scale blob detection"""
    # Multi-scale blob detection for circular structures
    scales = [1, 2, 4, 8]
    enhanced = np.zeros_like(image)
    
    for scale in scales:
        sigma = scale * sensitivity
        blurred = ndimage.gaussian_filter(image, sigma=sigma)
        dog = image - blurred
        enhanced += dog
    
    # Normalize enhancement
    enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
    
    # Combine with original image with weighting based on sensitivity
    return np.clip(image * (1 - sensitivity*0.3) + enhanced * sensitivity * 0.7, 0, 1)

def enhance_edges(image, strength=1.0):
    """Enhance edges for better nodule boundary detection"""
    # Sobel edge detection
    edges = filters.sobel(image)
    
    # Scale edges by strength factor
    scaled_edges = edges * strength
    
    # Add edges to original image
    return np.clip(image + scaled_edges, 0, 1)

def enhance_vessels(image):
    """Enhance vascular structures using Frangi filter"""
    try:
        from skimage.filters import frangi
        enhanced = frangi(image, sigmas=range(1, 10, 2), black_ridges=False)
        
        # Normalize enhancement
        enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
        
        # Blend with original
        return np.clip(image * 0.7 + enhanced * 0.3, 0, 1)
    except Exception as e:
        logger.error(f"Error in enhance_vessels: {str(e)}")
        # Fallback to a simpler enhancement if frangi fails
        return enhance_edges(image, 0.8)

def suppress_bones(image):
    """Suppress bone structures to highlight soft tissue"""
    # Identify high-intensity regions (likely bones)
    threshold = np.percentile(image, 90)
    mask = image > threshold
    
    # Reduce intensity in bone regions
    processed = image.copy()
    processed[mask] = processed[mask] * 0.6
    
    # Enhance remaining tissue
    processed = apply_clahe(processed, clip_limit=1.5)
    
    return processed

def apply_gamma(image, gamma):
    """Apply gamma correction"""
    return np.power(image, gamma)

def apply_sharpening(image, amount):
    """Apply unsharp masking for sharpening"""
    blurred = ndimage.gaussian_filter(image, sigma=1)
    return np.clip(image + (image - blurred) * amount, 0, 1)

def apply_denoising(image, strength):
    """Apply denoising"""
    if strength <= 0:
        return image
        
    try:
        # Try bilateral filter first (better preserves edges)
        denoised = restoration.denoise_bilateral(
            image, 
            sigma_color=strength*0.1,
            sigma_spatial=strength*2
        )
        return denoised
    except Exception:
        # Fallback to gaussian filter
        return ndimage.gaussian_filter(image, sigma=strength)

def create_figure(image):
    """Create and return a base64 encoded figure"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{image_base64}"

# Flask route handler
def handle_manual_analysis(file):
    """Process the uploaded file for manual analysis"""
    try:
        # Save file to temporary location
        filepath = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(filepath)
        
        # Load and enhance the image
        figure = load_and_enhance_xray(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return {'success': True, 'figure': figure}
        
    except Exception as e:
        logger.error(f"Error in handle_manual_analysis: {str(e)}")
        return {'error': str(e)}, 500

def handle_update_analysis(params):
    """Handle update analysis request"""
    try:
        figure = update_analysis(params)
        return {'success': True, 'figure': figure}
    except Exception as e:
        logger.error(f"Error in handle_update_analysis: {str(e)}")
        return {'error': str(e)}, 500

def update_figure(method="Standard", mode="Enhanced", contrast=1.0, brightness=0.0, 
                 enhancement=1.0, threshold=0.5, gamma=1.0, sharpness=1.0, denoise=0.0):
    """Update the image based on selected parameters"""
    global current_image
    
    try:
        # Get the current image
        if not hasattr(update_figure, 'current_image'):
            # If no image is stored, return None
            return None
            
        # Make a copy of the original image
        image = update_figure.current_image.copy()
        
        # Apply basic adjustments
        image = apply_basic_adjustments(image, contrast, brightness, gamma)
        
        # Apply selected method
        if method != "Standard":
            image = apply_method(image, method, enhancement)
        
        # Apply sharpening
        if sharpness != 1.0:
            image = apply_sharpening(image, sharpness)
            
        # Apply denoising
        if denoise > 0:
            image = apply_denoising(image, denoise)
            
        # Convert to base64
        return get_base64_image(image)
        
    except Exception as e:
        print(f"Error in update_figure: {str(e)}")
        return None

def apply_basic_adjustments(image, contrast, brightness, gamma):
    """Apply basic image adjustments"""
    # Convert to float for calculations
    image = image.astype(float)
    
    # Apply contrast
    image = image * contrast
    
    # Apply brightness
    image = image + (brightness * 255)
    
    # Apply gamma
    image = np.power(image/255, gamma) * 255
    
    # Clip values to valid range
    image = np.clip(image, 0, 255)
    
    return image.astype(np.uint8)

def apply_method(image, method, enhancement):
    """Apply selected enhancement method"""
    if method == "Nodule Detection":
        # Add nodule detection specific enhancement
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.addWeighted(image, 1 + enhancement, image, 0, 0)
        
    elif method == "Local Contrast":
        # Apply local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0 + enhancement, tileGridSize=(8,8))
        image = clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))[:,:,0]
        
    elif method == "Edge Enhancement":
        # Apply edge enhancement
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * enhancement
        image = cv2.filter2D(image, -1, kernel)
        
    elif method == "Vessel Enhancement":
        # Apply vessel enhancement
        image = cv2.GaussianBlur(image, (3,3), 0)
        image = cv2.addWeighted(image, 1 + enhancement, image, 0, 0)
    
    return image

def apply_sharpening(image, sharpness):
    """Apply image sharpening"""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpness
    return cv2.filter2D(image, -1, kernel)

def apply_denoising(image, denoise):
    """Apply denoising"""
    return cv2.fastNlMeansDenoisingColored(image, None, denoise * 10, denoise * 10, 7, 21)

def get_base64_image(image_array):
    """Convert image array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Add a method to set the current image
def set_current_image(image):
    """Set the current image for processing"""
    update_figure.current_image = image