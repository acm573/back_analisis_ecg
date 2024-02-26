from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

print('Cargando el modelo....')
modelo = load_model('hdsp_modelo_multiclase_ponderacion.keras')
print('Modelo cargado')

print('Cargando el datataset')
df = pd.read_csv('hdsp_segmentos_x.csv', header=None, index_col=False)

print(df.head())

y_pred = modelo.predict(df, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
print('Total de latidos: ', df.shape[0])
print('Clasificación de latidos: ', y_pred_classes)
