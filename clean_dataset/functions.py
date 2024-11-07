import cv2 as cv
import numpy as np
import os

def binarize(gray, umbral=200):
    umbral = 200
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


def write_image(cutted_image, output_folder, file, final_size=600):
    height, width = cutted_image.shape[:2]


    padding_vertical = max(0, (final_size - height) // 2)
    padding_horizontal = max(0, (final_size - width) // 2)

    imagen_512x512 = cv.copyMakeBorder(
        cutted_image,
        top=padding_vertical,
        bottom=padding_vertical + (final_size - height) % 2,
        left=padding_horizontal,
        right=padding_horizontal + (final_size - width) % 2, 
        borderType=cv.BORDER_CONSTANT,
        value=[255, 255, 255] 
    )

    output_path = os.path.join(output_folder, file)
    cv.imwrite(output_path, imagen_512x512)
