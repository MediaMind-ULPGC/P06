import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize

def enhance_fingerprint(image):
    gray_image = to_grayscale(image)
    enhanced_image = apply_frangi_filter(gray_image)
    enhanced_image = normalize_image(enhanced_image)
    enhanced_image = apply_adaptive_threshold(enhanced_image)
    return enhanced_image

def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_frangi_filter(image):
    return frangi(image)

def normalize_image(image):
    return (image * 255).astype(np.uint8)

def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def skeletonize_fingerprint(image):
    if image.dtype != np.bool:
        image = image > 127
    return skeletonize(image)