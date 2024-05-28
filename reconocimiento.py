import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Cargar el modelo de reconocimiento facial
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Función para realizar el reconocimiento facial en una imagen
def reconocimiento_facial(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (180, 180), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        if result[1] < 70:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, str(result[0]), (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, 'Desconocido', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return image

# Configurar la interfaz de Streamlit
st.title("Aplicación de Reconocimiento Facial")

option = st.radio("Selecciona una opción:", ("Cámara en Vivo", "Subir Imagen"))

if option == "Cámara en Vivo":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("No se puede acceder a la cámara.")
            break
        frame = reconocimiento_facial(frame)
        st.image(frame, channels="BGR", caption="Cámara en Vivo")
else:
    uploaded_file = st.file_uploader("Subir una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = reconocimiento_facial(image)
        st.image(image, channels="BGR", caption="Imagen Cargada")
