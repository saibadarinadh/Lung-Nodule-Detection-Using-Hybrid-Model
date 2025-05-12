import os
import io
import numpy as np
import cv2
import pydicom
from flask import Blueprint, request, jsonify, session, send_file, url_for
from PIL import Image

manual_analysis_bp = Blueprint('manual_analysis', __name__)

UPLOADS_DIR = 'uploads'
STATIC_DIR = 'static'
DICOM_PATH = os.path.join(UPLOADS_DIR, 'manual_analysis.dcm')

def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def apply_filters(img, params):
    # Basic
    brightness = params.get('brightness', 0)
    contrast = params.get('contrast', 1.0)
    gamma = params.get('gamma', 1.0)
    grayscale = params.get('grayscale', False)
    invert = params.get('invert', False)
    # Denoising
    median = params.get('median', 1)
    gaussian = params.get('gaussian', 0)
    bilateral_d = params.get('bilateral_d', 1)
    bilateral_sc = params.get('bilateral_sc', 10)
    bilateral_ss = params.get('bilateral_ss', 10)
    # Advanced
    clahe = params.get('clahe', 0)
    histeq = params.get('histeq', False)
    unsharp_amount = params.get('unsharp_amount', 0)
    unsharp_radius = params.get('unsharp_radius', 1)
    # Segmentation
    canny = params.get('canny', False)
    canny_threshold1 = params.get('canny_threshold1', 100)
    canny_threshold2 = params.get('canny_threshold2', 200)
    sobel = params.get('sobel', False)
    laplacian = params.get('laplacian', False)
    threshold = params.get('threshold', 128)
    adaptive_threshold = params.get('adaptive_threshold', False)
    # Morphology
    erosion = params.get('erosion', 1)
    dilation = params.get('dilation', 1)
    opening = params.get('opening', 1)
    closing = params.get('closing', 1)
    # Color
    channel = params.get('channel', 'all')

    out = img.copy().astype(np.float32)
    # Basic
    out = out * contrast + brightness
    out = np.clip(out, 0, 255)
    if gamma != 1.0:
        out = 255 * ((out / 255) ** (1.0 / gamma))
    out = np.clip(out, 0, 255)
    if grayscale:
        out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    if invert:
        out = 255 - out
    # Denoising
    if median > 1:
        out = cv2.medianBlur(out.astype(np.uint8), median)
    if gaussian > 0:
        out = cv2.GaussianBlur(out.astype(np.uint8), (gaussian*2+1, gaussian*2+1), 0)
    if bilateral_d > 1:
        out = cv2.bilateralFilter(out.astype(np.uint8), bilateral_d, bilateral_sc, bilateral_ss)
    # Advanced
    if clahe > 0:
        lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe_obj = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8,8))
        lab[:,:,0] = clahe_obj.apply(lab[:,:,0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if histeq:
        for i in range(3):
            out[:,:,i] = cv2.equalizeHist(out[:,:,i].astype(np.uint8))
    if unsharp_amount > 0:
        blurred = cv2.GaussianBlur(out.astype(np.uint8), (unsharp_radius*2+1, unsharp_radius*2+1), 0)
        out = cv2.addWeighted(out.astype(np.uint8), 1+unsharp_amount, blurred, -unsharp_amount, 0)
    # Segmentation
    if canny:
        edges = cv2.Canny(out.astype(np.uint8), canny_threshold1, canny_threshold2)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if sobel:
        sobelx = cv2.Sobel(out, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(out, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    if laplacian:
        lap = cv2.Laplacian(out, cv2.CV_64F)
        lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
    if adaptive_threshold:
        gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    elif threshold != 128:
        gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    # Morphology
    if erosion > 1:
        out = cv2.erode(out, np.ones((erosion,erosion), np.uint8), iterations=1)
    if dilation > 1:
        out = cv2.dilate(out, np.ones((dilation,dilation), np.uint8), iterations=1)
    if opening > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((opening,opening), np.uint8))
    if closing > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((closing,closing), np.uint8))
    # Channel selector
    if channel in ['r', 'g', 'b']:
        idx = {'r':0, 'g':1, 'b':2}[channel]
        ch = out[:,:,idx]
        out = np.zeros_like(out)
        out[:,:,idx] = ch
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def get_current_dicom_path():
    # Always get the last uploaded DICOM from the session
    return session.get('last_dicom_path', None)

@manual_analysis_bp.route('/api/manual-analysis-process', methods=['POST'])
def manual_analysis_process():
    dicom_path = get_current_dicom_path()
    if not dicom_path or not os.path.exists(dicom_path):
        return jsonify({'error': 'No DICOM image found. Please upload on the home page.'}), 400
    img = load_dicom_image(dicom_path)
    params = request.get_json()
    print("Received filter params:", params)
    processed = apply_filters(img, params)
    # Save processed image to a temp file
    out_path = os.path.join(STATIC_DIR, 'processed_image.png')
    Image.fromarray(processed).save(out_path)
    # Also save the original image for display
    orig_path = os.path.join(STATIC_DIR, 'original_image.png')
    Image.fromarray(img).save(orig_path)
    return jsonify({'processed_image_url': url_for('static', filename='processed_image.png')})

# You should also serve the original image as PNG for the viewer
@manual_analysis_bp.route('/api/manual-analysis-original')
def manual_analysis_original():
    img = load_dicom_image(DICOM_PATH)
    out_path = os.path.join(STATIC_DIR, 'original_image.png')
    Image.fromarray(img).save(out_path)
    return send_file(out_path, mimetype='image/png')

@manual_analysis_bp.route('/api/manual-analysis-upload', methods=['POST'])
def manual_analysis_upload():
    file = request.files.get('file')
    if not file or not file.filename.endswith('.dcm'):
        return jsonify({'error': 'No DICOM file uploaded'}), 400

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    file.save(DICOM_PATH)

    img = load_dicom_image(DICOM_PATH)
    orig_path = os.path.join(STATIC_DIR, 'original_image.png')
    Image.fromarray(img).save(orig_path)
    proc_path = os.path.join(STATIC_DIR, 'processed_image.png')
    Image.fromarray(img).save(proc_path)

    return jsonify({
        'original_image_url': '/static/original_image.png',
        'processed_image_url': '/static/processed_image.png'
    }) 