import procesar_imagen as proci
import procesar_segmentos as procs
import procesar_modelos as procm
from PIL import Image
from io import BytesIO
import base64


print('Evaluando el proceso de procesar la imagen')

ruta_imagen = '/Users/acm/Documents/Cursos/Health/Final/codigo/back_analisis_ecg/vector_010.bmp'

#carga la imagen
imagen = Image.open(ruta_imagen)

#convierte la imagen a bytes
buffer = BytesIO()
imagen.save(buffer, format='BMP')
img_bytes = buffer.getvalue()

#codifica a base64
img_base64 = base64.b64encode(img_bytes)

img_str = img_base64.decode('utf-8')

##Llamadas a las funciones para procesar imagen
## y procesar segmentos

ecg_vector = proci.procesar_imagen(img_str)

#Obtiene los latidos segmentados
df = procs.procesar_extraer_segmentos(ecg_vector)

#procesa la clasificacion
print(procm.clasificar_latidos(df))