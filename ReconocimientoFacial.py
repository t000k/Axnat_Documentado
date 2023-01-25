import cv2
import os
import serial
import time

ser = serial.Serial('COM3', 9600, timeout=1) #aqui vamos a inicializar un puerto serial que en este caso es el COM3
time.sleep(2) #esperamos 2 segundos

ruta = 'C:/Users/cisne/OneDrive/Escritorio/Reconocimiento Facial/Data' #ruta donde se encuentran almacenados los rostros de las personas
imagenPersona = os.listdir(ruta) #hacemos un listado de las carpetas que se encuentran en la ruta
print('personas=',imagenPersona) #imprimimos las carpetas de la ruta, es decir los usuarios
 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
reconocer_rostro = cv2.face.LBPHFaceRecognizer_create() #establezco el metodo que voy a usar para el reconocimiento facial

# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
reconocer_rostro.read('modeloLBPHFace.xml') #leemos el modelo almacenado previamente

captura = cv2.VideoCapture(0,cv2.CAP_DSHOW) #vamos a realizar el reconocimiento en una captura de video en tiempo real
#cap = cv2.VideoCapture('Video.mp4') 
clasificacion = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #vamos a leer el clasificador de rostros como en el programa inicial

while True:
	ret,frame = captura.read() #vamos a leer el video en tiempo real en fotogramas y la almacenamos en esta variable
	if ret == False: break #si no captura nada simplemente no comienza con el proceso de reconocer
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #cambiamos las imagenes a escalas de grises
	auxFrame = gray.copy()
	
	faces = clasificacion.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC) #redimensionamos las capturas de rostros a 150x150 para poder procesarlas
		result = reconocer_rostro.predict(rostro) #este es el proceso donde se esta reconociendo el rostro

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		'''
		# EigenFaces
		if result[1] < 5700:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
		# FisherFace
		if result[1] < 500:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		'''
		# LBPHFace
		
		if result[1] < 70: #si el grado de concidencia de las capturas acuales son las suficientes para conincidir con los modelos almacenados entonces:
			cv2.putText(frame,'{}'.format(imagenPersona[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA) #colocamos una etiqueta correspondiente a imagenPersona
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #dibujamos un rectangulo al rededor del rostro reconocido actualmente
			ser.write(b'P') #escribimos hacia el arduino el dato 'P' por el puerto serial
			
		else: #si no
			cv2.putText(frame,'No es alumno',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA) #en el rostro detectado actualmente coloca la etiqueta de no es alumno
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) #dibuja un rectangulo color rojo en el rostro detectado actualmente
			ser.write(b'N') #escribimos hacia el arduino el dato el dato 'N' por el puerto serial
		
	
		
	

		
		
	
	cv2.imshow('Reconocimiento facial',frame) #nombre de la ventana
	k = cv2.waitKey(1)
	if k == 27: #terminamos el proceso al presionar la tecla ESC
		break

ser.close() 
captura.release()
cv2.destroyAllWindows()
