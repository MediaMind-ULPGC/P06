import os
import cv2 as cv
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
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image

def apply_frangi_filter(image):
    return frangi(image)

def normalize_image(image):
    return (image * 255).astype(np.uint8)

def apply_adaptive_threshold(image):
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

def reduce_noise(image):
    return cv.GaussianBlur(image, (5, 5), 0)

def equalize_histogram(image):
    return cv.equalizeHist(image)

def skeletonize_fingerprint(image):
    if image.dtype != np.bool:
        image = image > 127
    return skeletonize(image)

def build_dataset_frangi(sample_path, subfolders, dataset_path):
    for subfolder in subfolders:
        path = os.listdir(sample_path+'\\'+subfolder)
        output_folder = dataset_path + '/' + subfolder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for file in path:
            route = sample_path+'\\'+subfolder+'\\'+file
            img = cv.imread(route, 0)
        
            noisy_reduced_image = reduce_noise(img)

            equalized_image = equalize_histogram(noisy_reduced_image)
            enhanced_image = enhance_fingerprint(equalized_image)
            skeleton_image = skeletonize_fingerprint(enhanced_image)
            skeleton_image = (skeleton_image * 255).astype(np.uint8)

            output_path = os.path.join(output_folder, file)
            
            imagen_resized = cv.resize(skeleton_image, (224, 224), interpolation=cv.INTER_LINEAR)
    
            cv.imwrite(output_path, imagen_resized)