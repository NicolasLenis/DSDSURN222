import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import os

# Función para calcular el EAR de un ojo
def aspecto_del_ojos(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Función para calcular la distancia entre el labio superior e inferior
def lip_distancia(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distancia = abs(top_mean[1] - low_mean[1])
    return distancia

# Carga del clasificador Haar y el predictor de Dlib
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Cargar la imagen
ruta_imagen = 'P2_1.jpg'  # Reemplaza con la ruta de tu imagen
frame = cv2.imread(ruta_imagen)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detección de rostro
rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Constantes para determinar insomnio
EYE_AR_THRESH = 0.2  # Umbral de EAR para ojos cerrados

for (x, y, w, h) in rects:
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    
    # Obtener puntos de referencia facial
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Calcular EAR para ambos ojos
    leftEye = shape[42:48]  # Puntos 42 a 48 para el ojo izquierdo
    rightEye = shape[36:42]  # Puntos 36 a 42 para el ojo derecho
    leftEAR = aspecto_del_ojos(leftEye)
    rightEAR = aspecto_del_ojos(rightEye)
    ear = (leftEAR + rightEAR) / 2.0  # Promedio de ambos EAR

    # Determinar si los valores indican insomnio y asignar el mensaje correspondiente
    if ear >= EYE_AR_THRESH:

        mensaje_estado = "Sin insomnio detectado"  # Mensaje por defecto
        color_mensaje = (0, 255, 0)  # Verde por defecto
    if ear < EYE_AR_THRESH:
        mensaje_estado = "Insomnio: Ojos cerrados"
        color_mensaje = (0, 0, 255)  # Rojo para alerta

    # Mostrar el mensaje en la imagen
    cv2.putText(frame, mensaje_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mensaje, 2)

# Proporciones deseadas para mostrar la imagen
width = 500  # Ancho deseado
height = int((frame.shape[0] * width) / frame.shape[1])  # Altura ajustada proporcionalmente
frame = cv2.resize(frame, (width, height))

# Define la nueva ruta de la imagen
nombre_archivo = 'P2_1.jpg'  # Asume que el archivo se llama 'P28_1.jpg'
nueva_carpeta = 'C:\\Users\\Lenis\\Desktop\\Resultado Final\\dlib\\0.2'# Actualiza esta ruta a la ubicación deseada
nueva_ruta_imagen = os.path.join(nueva_carpeta, nombre_archivo)  # Ruta completa del nuevo archivo

# Mostrar la imagen
cv2.imshow("Insomnio detectado", frame)
cv2.waitKey(0)

# Guarda la imagen en la carpeta especificada con el nombre modificado
if not os.path.exists(nueva_carpeta):
    os.makedirs(nueva_carpeta)
cv2.imwrite(nueva_ruta_imagen, frame)

cv2.destroyAllWindows()
