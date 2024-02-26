from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np





def filtrar_latidos_anomalia(df_latidos, resultados_clasificacion):
    # Convertir el vector de clasificación en una serie de pandas para usar su indexación booleana
    serie_clasificacion = pd.Series(resultados_clasificacion)
    # Filtrar los latidos donde la clasificación es 1 (anomalía)
    df_latidos_anomalia = df_latidos[serie_clasificacion == 1]
    return df_latidos_anomalia



def clasificacion_binaria(df):
    print('Cargando el modelo....')
    modelo = load_model('hdsp_modelo_binario.h5')
    print('Modelo cargado')

    y_pred = modelo.predict(df, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print('Total de latidos: ', df.shape[0])
    print('Clasificación de latidos: ', y_pred_classes)


    return y_pred_classes


def clasificacion_anomalia(df):
    print('Cargando el modelo anomalias....')
    modelo = load_model('hdsp_modelo_anomalia.h5')
    print('Modelo cargado anomalias')

    y_pred = modelo.predict(df, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print('Total de latidos: ', df.shape[0])
    print('Clasificación de latidos: ', y_pred_classes)


    return y_pred_classes


def actualizar_resultados_clasificacion(resultados_clasificacion, resultados_anomalia):
    # Índice para rastrear la posición en resultados_anomalia
    indice_anomalia = 0
    
    # Iterar sobre resultados_clasificacion por su índice y valor
    for i, valor in enumerate(resultados_clasificacion):
        # Si el valor es 1 (anomalía), actualizamos con el valor correspondiente de resultados_anomalia
        if valor == 1:
            resultados_clasificacion[i] = resultados_anomalia[indice_anomalia]
            indice_anomalia += 1  # Mover al siguiente valor en resultados_anomalia
    
    # Aseguramos que todos los valores fueron actualizados correctamente
    assert indice_anomalia == len(resultados_anomalia), "La cantidad de anomalías no coincide con los resultados de clasificación específica."
    
    return resultados_clasificacion

# Ejemplo de uso:
# resultados_clasificacion = [0, 1, 0, 1, 0]  # Vector original de clasificación
# resultados_anomalia = [2, 3]  # Clasificación específica de las anomalías
# resultados_clasificacion_actualizados = actualizar_resultados_clasificacion(resultados_clasificacion, resultados_anomalia)
# print(resultados_clasificacion_actualizados)



def clasificar_latidos(df):

    #realiza la clasificación binaria como primer paso
    pred_binaria = clasificacion_binaria(df)

    #se necesita separar a los latidos que son diferente de cero
    df_latidos_anomalia = filtrar_latidos_anomalia(df, pred_binaria)

    # Mostrar el dataframe resultante
    print(df_latidos_anomalia)
    print('forma -> ', df_latidos_anomalia.shape)

    if df_latidos_anomalia.shape[0] != 0:
        print('hay anomalias')

        pred_anomalia = clasificacion_anomalia(df_latidos_anomalia)

        print('Clasificacion anomalia: ', pred_anomalia)

        clasificacion_final = actualizar_resultados_clasificacion(pred_binaria, pred_anomalia)


    else:
        print('no hubo anomalias')
        clasificacion_final = pred_binaria


    return clasificacion_final