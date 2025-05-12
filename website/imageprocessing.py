import cv2
import numpy as np

def apply_basic_adjustments(image, brightness=0, contrast=1.0, gamma=1.0):
    """Apply basic image adjustments with clamping to prevent overexposure"""
    image = image.astype(float)
    # Clamp brightness and contrast
    brightness = np.clip(brightness, -50, 50)
    contrast = np.clip(contrast, 0.5, 2.0)
    image = image * contrast
    image = image + (brightness * 2.5)  # scale brightness for finer control
    image = np.power(np.clip(image/255, 0, 1), gamma) * 255
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
    """Apply contrast enhancement"""
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        image = clahe.apply(image)
    if sharpen > 0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpen
        image = cv2.filter2D(image, -1, kernel)
    return image

def apply_histogram_equalization(image):
    """Apply global histogram equalization"""
    return cv2.equalizeHist(image)

def apply_adaptive_threshold(image, block_size=11, C=2):
    """Apply adaptive thresholding for segmentation"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block_size, C)

def apply_sobel(image):
    """Apply Sobel edge detection"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.clip(sobel, 0, 255))

def apply_laplacian(image):
    """Apply Laplacian edge detection"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))

def apply_blob_detection(image, min_area=30, max_area=500):
    """Detect blobs (potential nodules) in the image"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = 0.5
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints

def apply_edge_detection(image, canny=False, sobel=False, laplacian=False, threshold=0):
    """Apply edge detection"""
    result = image.copy()
    if canny:
        result = cv2.Canny(image, 100, 200)
    if sobel:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobelx, sobely)
    if laplacian:
        result = cv2.Laplacian(image, cv2.CV_64F)
    if threshold > 0:
        _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return result

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