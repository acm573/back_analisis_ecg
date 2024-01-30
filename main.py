# Librerías
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64

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
        # Procesar la imagen con funciónA
        vector_doble = funcionA(image_data)
        # Analizar la información con funciónB
        result = funcionB(vector_doble)
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

# Función que realizará la extracción del trazo de ECG directo de la imagen
def funcionA(image_data):

    #colocar aquí la lógica para extrar el trazo

    #retornar el vector doble de 10 segundos de la imagen
    pass


# Funcion que evalua el segmento de trazo ECG con el modelo
def funcionB(vector_doble):

    #llamar el modelo para la clasificacion de latidos


    #simula una respuesta de analisis para enviar respuesta al front
    #y guardar información en MongoDB
    totalLatidos = 4
    latidos = [
        { 'tipo': 'Normal', 'latido': [1, 2, 3, 4, 5] },
        { 'tipo': 'Atípico', 'latido': [6, 7, 8, 9, 10] },
        { 'tipo': 'Normal', 'latido': [1, 2, 3, 4, 5] },
        { 'tipo': 'Atípico', 'latido': [6, 7, 8, 9, 10] }
    ]
    return {'totalLatidos' : totalLatidos, 'latidos': latidos}


# se creó un entorno para el desarrollo del back de análisis
# conda create --name analisis_ecg python=3.10
# conda activate analisis_ecg
# conda install -c conda-forge fastapi uvicorn


# para ejecutarlo: uvicorn main:app --reload
# desde: /Users/acm/Documents/Cursos/Health/Final/codigo/fastapi

