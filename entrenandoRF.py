import cv2
import os
import numpy as np

ruta = 'C:/Users/cisne/OneDrive/Escritorio/Reconocimiento Facial/Data' #Cambia a la ruta donde hayas almacenado Data
lista_personas = os.listdir(ruta) #hace un listado de las carpetas de usuarios
print('Lista de personas: ', lista_personas) #imprime lista_personas

labels = [] #etiqueta correspondiente a cada imagen y persona
facesData = [] #aqui se almacenará cada una de las imagenes de los rostros
label = 0 #este label nos ayuda a contar las imagenes y con ello determinar cue imagenes le toca a la persona

for dirNombre in lista_personas:
	personaRuta = ruta + '/' + dirNombre #vamos a leer cada una de las carpetas dentro de data
	print('Leyendo las imágenes') #imprimir este letrero para saber en que parte de el proceso estamos

	for fileName in os.listdir(personaRuta): #Aquí leeremos todas las imágenes correspondientes a cada rostro. En la línea 18 estamos imprimiendo el nombre de la carpeta y la imagen.
		print('Rostros: ', dirNombre + '/' + fileName) # //
		labels.append(label) #aqui vamos añadiendo la etiqueta de cada imagen
		facesData.append(cv2.imread(personaRuta+'/'+fileName,0)) #en este array vamos a aañadir cada imagen del rostro
		#image = cv2.imread(personaRuta+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1 #por cada vez que se termine de almacenar los rostros y etiquetas de la carpeta vamos a aumentarle +1 a label

#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0)) estas lineas comentadas son si queremos ver el numero de etiquetas almacenadas
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
reconocer_rostro = cv2.face.LBPHFaceRecognizer_create() #este es un metodo desarrollado por openCV para poder reconocer los rostros, utilizaremos el metodo LBPHF pya que es el mejor en cuanto reconocer un rostro

# Entrenando el reconocedor de rostros
print("Entrenando...") #imprimir este letrero para saber en que parte del proceso estamos
reconocer_rostro.train(facesData, np.array(labels)) #esta es la linea que entrena al metodo LBPHF para poder reconocer los rostros pide dos parametros. El primero es para especificar el array donde estan contenidos los rostros y el segundo parametro corresponde a las etiquetas

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
reconocer_rostro.write('modeloLBPHFace.xml') #almacenamos el entrenamiento obtenido
print("Modelo almacenado...") #indicamos que termino el proceso