#PROYECTO DE DETECCIÓN SOMNOLENCIA CON DLIB
#ESTUDIANTES:
#NICOLAS LENIS SANCHEZ
#OMAR ALFONSO GALVIS CAMARON

#Librerias necesarias para el funcionamiento del código
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from pygame import mixer
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

#Agregamos alarma para el aviso de la deteccion de somnolencia
mixer.init()
mixer.music.load('alarm.wav')

#función para calcular el EAR del ojo
def aspecto_del_ojos(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    ear = (A + B) / (2.0 * C)
    return ear

#Función para calcular el EAR promedio de ambos ojos
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_eye = shape[lStart:lEnd]
    right_Eye = shape[rStart:rEnd]

    leftEAR = aspecto_del_ojos( left_eye)
    rightEAR = aspecto_del_ojos(right_Eye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, left_eye, right_Eye)

#Se establecen argumentos y variables generales para la detección
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--camara", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.2 #limite de EAR
EYE_AR_CONSEC_FRAMES = 5 #tiempo limite para ojos cerrados
estado_1 = False
contador = 0

#Se carga el clasificador Haar para la detección de rostros con OpenCV
print("-> cargando el sistema...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Se carga el predictor de los 68 puntos faciales que tiene la librería Dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Se inicia la captura de video y se espera un 1 segundo para que la cámara se active
print("-> iniciando camara")
vs = VideoStream(src=args["camara"]).start()
time.sleep(1.0)

while True:
    #Se captura y redimensiona el frame
    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    #Se convierte a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Se Detecta el rostro
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    #Se inicia un for para detectar rostros y hacer su respectivo análisis
    for (x, y, w, h) in rects:

        #Se calcula el EAR, luego se calcula la distancia de labios y se dibuja los contornos 
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ojo = final_ear(shape)
        ear = ojo[0]
        leftEye = ojo[1]
        rightEye = ojo[2]

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Si el Ear es muy bajo, se inicia un conteo para tener en cuenta cuanto tiempo se tienen los ojos cerrados
        if ear < EYE_AR_THRESH:
            contador += 1
            #si este tiempo supera el permitido se inicia el aviso de somnolencia
            if contador >= EYE_AR_CONSEC_FRAMES:
                if estado_1 == False:
                    estado_1 = True
                    mixer.music.play()
                #Se da alerta en pantalla
                cv2.putText(frame, "somnoliento!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            contador = 0
            estado_1 = False
                
        #Se muestran en pantalla los valores en tiempo real del EAR
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    apagado = cv2.waitKey(1) & 0xFF
    #Se permite cerrar el programa con la tecla q si se desea
    if apagado == ord("q"):
        break

#Finalmente si se cierra el programa se detienen todas las ventanas y el stream
cv2.destroyAllWindows()
vs.stop()
