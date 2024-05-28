import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

dataPath = 'D:/Universidad/6/Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

def load_model():
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')
    return face_recognizer
face_recognizer=load_model()

def load_image(image_file):
    img = Image.open(image_file)
    return img

def detect_faces(image):
    # Convert the image to a numpy array
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV pre-trained face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def draw_faces(image, faces,frame, only=False):

    # Convert the image to a numpy array
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    

    for (x, y, w, h) in faces:
            
        if only:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:    
            
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(image_np,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

            if result[1] < 70:
                cv2.putText(image_np,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(image_np, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(image_np,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(image_np, (x,y),(x+w,y+h),(0,0,255),2)
    return image_np

st.set_page_config(page_title="Reconocimiento Facial en Vivo", page_icon=":camera:", layout="wide")
st.title("📸 Aplicación de Reconocimiento de Rostros en Vivo")

st.sidebar.title("Opciones")
image_source = st.sidebar.selectbox("Seleccione la fuente de la imagen", ["Subir Imagen", "Cámara en Vivo"])

image = None
stop_camera = False

if "stop_camera" not in st.session_state:
    st.session_state["stop_camera"] = False

if "reset" not in st.session_state:
    st.session_state["reset"] = False

if image_source == "Subir Imagen":
    image_file = st.sidebar.file_uploader("Cargue una imagen", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image = load_image(image_file)
else:
    st.write("### La cámara en vivo está activada. Por favor, espere un momento para que se muestre el video.")
    stop_button = st.button("Detener Cámara")
    if stop_button:
        st.session_state["stop_camera"] = True

    reset_button = st.button("Activar camara")
    if reset_button:
        st.session_state["reset"] = True
        st.session_state["stop_camera"] = False
        st.experimental_rerun()

    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    while not st.session_state["stop_camera"]:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara")
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detect_faces(Image.fromarray(frame))
        
        # Draw faces
        frame_with_faces = draw_faces(Image.fromarray(frame), faces,frame)
        
        # Update the Streamlit image element
        frame_window.image(frame_with_faces, channels="RGB")

    cap.release()

if image is not None:
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Detect faces
    faces = detect_faces(image)
    st.write(f"### Se detectaron {len(faces)} rostro(s) en la imagen.")

    # Draw faces
    if len(faces) > 0:
        result_image = draw_faces(image, faces)
        st.image(result_image, caption='Imagen con rostros detectados', use_column_width=True)
else:
    if image_source == "Subir Imagen":
        st.write("### Por favor, cargue una imagen para detectar rostros.")
