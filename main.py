# Librerías
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import procesar_imagen as proci
import procesar_segmentos as procs
import procesar_modelos as procm
import matplotlib.pyplot as plt


# Definir modelos para FastAPI
class RequestModel(BaseModel):
    imageBase64: str

class BeatInfo(BaseModel):
    tipo: str
    latido: list

class ResponseModel(BaseModel):
    totalLatidos: int
    latidos: list[BeatInfo]
    imageBase64: str

# Inicializar la aplicación FastAPI
app = FastAPI()

# Endpoint que recibe la imagen para su procesamiento
@app.post("/analyze_ecg/", response_model=ResponseModel)
async def analyze_ecg(request: RequestModel):
    try:
        #confirma la llamada
        print('Llamada a funcion analyze_ecg')
        # Convertir la imagen base64 a un formato utilizable
        image_data = decode_image(request.imageBase64)

        #Obtiene el vector del ecg de la seccion
        ecg_vector = proci.procesar_imagen(request.imageBase64)
     
        #Obtiene los latidos segmentados
        df = procs.procesar_extraer_segmentos(ecg_vector)

        #implemnenta la clasificación de los latidos
        clasificacion = procm.clasificar_latidos(df)

        print('clasificacion_final: ', clasificacion)

        # Analizar la información con funciónB
        result = funcionB(df, clasificacion)
        # Construir la respuesta
        response = ResponseModel(totalLatidos=result['totalLatidos'],
                                 latidos=result['latidos'],
                                 imageBase64=request.imageBase64)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Función para decodificar la imagen de base64 a bytes
def decode_image(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    return image_bytes



# Funcion que evalua el segmento de trazo ECG con el modelo
def funcionB(df, clasificacion):

    #itera el dataframe para generar la respuesta
    tipos = ['[N] - Normal','[S] - Supraventricular','[V] - Ventricular','[F] - Fusión','[Q] - Desconocido']

    latidos = []

    for i in range(df.shape[0]):
        latidos.append(
            {
                'tipo': tipos[clasificacion[i]],
                'latido': df.iloc[i]
            }
            )

    #print(latidos)

#    totalLatidos = 5
#    latidos = [
#        { 'tipo': 'Normal (N)', 'latido': [1, 2, 3, 4, 5] },
#        { 'tipo': 'Supraventricular (S)', 'latido': [6, 7, 8, 9, 10] },
#        { 'tipo': 'Ventricular (V)', 'latido': [1, 2, 3, 4, 5] },
 #       { 'tipo': 'Unknown (Q)', 'latido': [6, 7, 8, 9, 10] },
 #       { 'tipo': 'Fusion Beat (F)', 'latido': [6, 7, 8, 9, 10] }
 #   ]
    return {'totalLatidos' : df.shape[0], 'latidos': latidos}


# se creó un entorno para el desarrollo del back de análisis
# conda create --name analisis_ecg python=3.10
# conda activate analisis_ecg
# conda install -c conda-forge fastapi uvicorn


# para ejecutarlo: uvicorn main:app --reload
# desde: /Users/acm/Documents/Cursos/Health/Final/codigo/fastapi











