import cv2
import os
import numpy as np

dataPath = 'D:/Universidad/6/Data'
peopleList = os.listdir(dataPath)
print('lista de persona', peopleList)

labels = []
facesData = []
label = 0

desired_width = 180
desired_height = 180

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Persona: ', nameDir + '/' + fileName)
        labels.append(label)
        # Leer la imagen y redimensionarla al tamaño deseado
        image = cv2.imread(os.path.join(personPath, fileName), 0)
        resized_image = cv2.resize(image, (desired_width, desired_height))
        facesData.append(resized_image)
    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")