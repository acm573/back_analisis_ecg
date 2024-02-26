#Funciones para obtener los segmentos de latidos de la imagen
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pandas as pd

#función para remuestrear la información a 125 Hz
def remuestreo(vector_original, fs_original, fs_nueva):
    # Calcular la cantidad de muestras en el nuevo vector
    n_muestras_original = len(vector_original)
    n_muestras_nueva = int(n_muestras_original * fs_nueva / fs_original)

    # Remuestrear el vector
    vector_remuestreado = resample(vector_original, n_muestras_nueva)

    return vector_remuestreado

#Funcion para normalizar el vector
def normalizar_vector(vector):
    minimo = np.min(vector)
    maximo = np.max(vector)
    vector_normalizado = (vector - minimo) / (maximo - minimo)
    return vector_normalizado

def comparar(a, b):
    return a > b

def media_latidos(indices):
    # Calcular las diferencias entre los índices consecutivos de los máximos
    diferencias = np.diff(indices)

    # Calcular la media de las diferencias
    media_diferencias = np.mean(diferencias)
    
    return media_diferencias

def encontrar_maximos(vector, umbral, distancia_minima):
    # Encontrar los índices de los máximos locales que superan el umbral
    indices_maximos = argrelextrema(vector, comparar, order=distancia_minima)

    # Filtrar los máximos locales para asegurar que superen el umbral
    indices_filtrados = [indice for indice in indices_maximos[0] if vector[indice] > umbral]

    return np.array(indices_filtrados)


def grafica_normalizado_latidos(vector_norm, indices):
    plt.figure(figsize=(15, 5))
    # Graficar el vector normalizado
    plt.plot(vector_norm)

    # Graficar los máximos locales encontrados
    plt.plot(indices, vector_norm[indices], 'ro')

    # Configurar la gráfica
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.title('Vector normalizado y máximos locales')

    # Mostrar la gráfica
    plt.show()

#Función que extrae los segmentos
def extraer_segmentos_10seg(vector, indices_maximos, media):
    segmentos = []
    ventana = int(media / 2)
    
    for indice in indices_maximos:
        inicio = indice - ventana
        fin = indice + ventana
        
        # Verificar si hay suficientes muestras a la izquierda y a la derecha del índice
        if inicio >= 0 and fin < len(vector):
            segmento = vector[inicio:fin]
            segmentos.append(segmento)
    
    return segmentos


def ajustar_longitud(segmentos, longitud_objetivo):
    segmentos_ajustados = []
    
    for segmento in segmentos:
        if len(segmento) < longitud_objetivo:
            # Rellenar con ceros si el segmento es más corto que la longitud objetivo
            segmento_ajustado = np.pad(segmento, (0, longitud_objetivo - len(segmento)), 'constant')
        else:
            # Si el segmento es más largo que la longitud objetivo, tomar los primeros 187 valores
            segmento_ajustado = segmento[:longitud_objetivo]
        
        segmentos_ajustados.append(segmento_ajustado)
    
    return segmentos_ajustados


def procesar_extraer_segmentos(ecg_vector):

    graficar = False

    #calcula el muestreo del vector que se ha extraido
    muestreo = len(ecg_vector) / 10.0

    print(muestreo)

    #se ajusta el muestreo del vector
    vector_125 = remuestreo(ecg_vector, muestreo, 125)
    
    if graficar == True:
        print('Longitud del vector: ', len(vector_125))

    #se normaliza el vector
    vector_norm_original = normalizar_vector(vector_125) 

    if graficar == True:
        plt.plot(vector_norm_original)
        plt.show()
        print(vector_norm_original[0:10])

    #deriva el vector
    vector_der = np.gradient(vector_norm_original)

    #elimina valores negativos
    vector_sin_ceros = np.clip(vector_der, 0, None)

    #normaliza el vector
    vector_norm = normalizar_vector(vector_sin_ceros)

    #distancia_minima = 10  # Cambia este valor según la distancia mínima deseada entre máximos
    indices_maximos = encontrar_maximos(vector_norm, 0.2, 50)

    #calcula la media entre latidos
    media = media_latidos(indices_maximos)

    print(indices_maximos)
    print(media)

    if graficar == True:
        print("Gráfica de latidos con vector derivado")
        grafica_normalizado_latidos(vector_norm_original, indices_maximos)

    #extrae segmentos en los 10 segundos
    segmentos_extraidos = extraer_segmentos_10seg(vector_norm_original, indices_maximos, media)

    #ajusta longitud de los segmentos
    segmentos_ajustados = ajustar_longitud(segmentos_extraidos, 187)
        
    #genera el dataframe de salida para el modelo
    df_segmentos = pd.DataFrame(segmentos_ajustados)

    if graficar == True:
        plt.plot(segmentos_ajustados[0])
        plt.show()

    print('Esta es salida --> ',df_segmentos.head())

    plt.plot(df_segmentos)
    plt.show()

    return df_segmentos