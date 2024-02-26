#Contiene los métodos para el procesamiento de la imagen
#que se recibe desde el front por parte del usuario
#para su análsis

from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2


#Función para obtener de base64 a imagen en escala se grises
def procesar_imagen_grises(img_str):
    #Se obtiene la imagen a partir de la cadena de base64 que se ha recibido
    imagen = base64.b64decode(img_str)
    imagen = Image.open(BytesIO(imagen))

    #Se ajusta la imagen a una escala de gris
    imagen_gris = imagen.convert('L')    

    return imagen_gris


#Funcion para limpiar la imagen
def procesar_imagen_gris_limpia(imagen_gris):
    threshold = 100
    imagen_gris_limpia = imagen_gris.point(lambda x: 0 if x < threshold else 255, '1')

    return imagen_gris_limpia


#Funcion para procesar la extracción del vector
def procesar_extraccion(seccion):
    puntos_blancos = np.where(seccion == 255)
    x_coords = puntos_blancos[1]
    y_coords = puntos_blancos[0]
    height, _ = seccion.shape

    unique_x_coords = np.unique(x_coords)
    y_values = np.zeros(unique_x_coords.shape, dtype=np.float64)

    for i, x in enumerate(unique_x_coords):
        y_for_x = y_coords[np.where(x_coords == x)]
        y_values[i] = np.max(y_for_x)

    y_values_inverted = height - y_values
    ecg_vector = y_values_inverted.astype(np.float64)

    return ecg_vector


#Funcion principal de procesamiento
def procesar_imagen(img_str):
    graficar = False

    #obtiene imagen en escala de grises
    imagen_gris = procesar_imagen_grises(img_str)
    
    if graficar == True:
        plt.figure(figsize=(10, 4))
        plt.imshow(imagen_gris, cmap = 'gray')
        plt.title('Imagen en escala de gris')
        plt.axis('off')

        plt.show()

    #se limpia la imagen
    imagen_gris_limpia = procesar_imagen_gris_limpia(imagen_gris)

    if graficar == True:
        #visualizamos la imagen en escala de gris
        plt.figure(figsize=(10, 4))
        plt.imshow(imagen_gris_limpia, cmap = 'gray')
        plt.title('Imagen en escala de gris')
        plt.axis('off')

        plt.show()

    #obtiene el negativo de la imagen
    imagen_invertida = np.array(imagen_gris_limpia).astype(np.uint8)
    imagen_invertida = Image.fromarray(255 - imagen_invertida)

    if graficar == True:
        #visualizamos la imagen en escala de gris
        plt.figure(figsize=(10, 4))
        plt.imshow(imagen_invertida, cmap = 'gray')
        plt.title('Imagen en escala de gris')
        plt.axis('off')

        plt.show()

    #Extrae la región de interes
    imagen_normalizada = np.array(imagen_invertida).astype(np.uint8)
    imagen_normalizada = cv2.normalize(imagen_normalizada, None, 0, 255, cv2.NORM_MINMAX)
    seccion = imagen_normalizada[417:507, 60:1000]

    if graficar == True:
        plt.figure(figsize=(10, 4))
        plt.imshow(seccion, cmap = 'gray')
        plt.title('Imagen en escala de gris')
        plt.axis('off')

        plt.show()

    #extrae el vector de puntos de la seccion de imagen
    ecg_vector = procesar_extraccion(seccion)

    if graficar == True:
        #Grafica el vector resultante
        plt.figure(figsize=(10, 4))
        plt.ylim(0, 100)
        plt.plot(ecg_vector)
        plt.title('Vector extraido')

        plt.show()

    return ecg_vector