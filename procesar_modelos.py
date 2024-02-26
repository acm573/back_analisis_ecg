from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np


def clasificacion_binaria(df):
    print('Cargando el modelo....')
    modelo = load_model('hdsp_modelo_binario.h5')
    #modelo = load_model('hdsp_modelo_multiclase_ponderacion.keras')
    print('Modelo cargado')

    y_pred = modelo.predict(df, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print('Total de latidos: ', df.shape[0])
    print('Clasificación de latidos: ', y_pred_classes)

    return y_pred_classes


def clasificar_latidos(df):

    #realiza la clasificación binaria como primer paso
    pred_binaria = clasificacion_binaria(df)

    return pred_binaria