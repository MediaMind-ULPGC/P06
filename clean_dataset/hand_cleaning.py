import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def binarize(gray, umbral=200):
    _, binary = cv.threshold(gray, umbral, 255, cv.THRESH_BINARY_INV)
    binary = binary / 255

    return binary

def calculate_values(histogram, total, percentage=0.9):
    acumulado_objetivo = total * percentage
    acumulado = 0
    inicio = None
    fin = None
    for i, valor in enumerate(histogram):
        acumulado += valor
        if acumulado >= acumulado_objetivo*0.05 and inicio is None:
            inicio = i
        if acumulado >= acumulado_objetivo:
            fin = i 
            break
    
    return inicio, fin

def cut_image(img, binary):
    histogram = np.sum(binary, axis=0)
    total = np.sum(binary)

    inicio_x, fin_x = calculate_values(histogram, total)
    
    histogram = np.sum(binary, axis=1)
    inicio_y, fin_y = calculate_values(histogram, total)

    cutted_image = img[inicio_y:fin_y, inicio_x:fin_x]
    return cutted_image


def write_image(cutted_image, output_folder, file, final_size=224):
    imagen_resized = cv.resize(cutted_image, (final_size, final_size), interpolation=cv.INTER_LINEAR)
    
    output_path = os.path.join(output_folder, file)
    gray = cv.cvtColor(imagen_resized, cv.COLOR_BGR2GRAY)
    
    cv.imwrite(output_path, gray)

def build_dataset(sample_path, subfolders, dataset_path):
    for subfolder in subfolders:
        path = os.listdir(sample_path+'\\'+subfolder)
        output_folder = dataset_path + '/' + subfolder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for file in path:
            route = sample_path+'\\'+subfolder+'\\'+file
            img = cv.imread(route)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            binary = binarize(gray)
            cutted_image = cut_image(img, binary)
            write_image(cutted_image, output_folder, file)