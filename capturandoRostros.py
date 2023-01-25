#librerias que estaremos utilizando:
import cv2 #libreria para usas OpenCV
import os  #libreria para poder ejecutar comandos del sistema
import imutils #libreria para procesamiento de imagenes
import tkinter as tk #libreria para poder desarrollar interfaces graficas
from tkinter import ttk #//
from tkinter import * #//



def persona(): #metodo persona
   name = str(caja_usuarios.get()) #este es la variable del nombre de la persona a registrar que obtenemos de un textbox
   nametext = 'C:/Users/cisne/OneDrive/Escritorio/Reconocimiento Facial/Data'+'/' + name #aqui creamos una variable string que haga referenci a una ruta exixtente en la maquina y la concatenamos con el nombre obtenido
   if not os.path.exists(nametext):     #
    print('Carpeta creada: ', nametext) #de la linea 14 a 16 creamos la carpeta a partir de la variable name text
    os.makedirs(nametext)               #en dichas lineas estamos haciendo uso de la libreria os para ejecutar comandos del sistema y crear carpetas
   
   captura = cv2.VideoCapture(0,cv2.CAP_DSHOW) #vamos a indicarle a opencv que inicie un video en vivo
#cap = cv2.VideoCapture('Video.mp4')  //tambien se puede hacer con un video

   clasificar = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #definimos que haarcascade usara openCV
   count = 0

   while True:


    ret, frame = captura.read() #leera los fotogramas obenidos de la variable captura 
    if ret == False: break
    frame =  imutils.resize(frame, width=640) #declaramos las dimensiones de la ventana donde se realizara la captura
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #utilizamos una escala de grises para poder usar el reconocimiento
    auxFrame = frame.copy() #variable auxiliar

    faces = clasificar.detectMultiScale(gray,1.3,5) #parametro que se debe definir para poder usar el haarcascade con la escala definida

    for (x,y,w,h) in faces: #este for es para crear tambien los rectangulos en el rostro
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #vamos a dibujar un rectangulo al rededor del rotro reconocido
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) # redimensionamos las capturas a 150 x 150
        cv2.imwrite( nametext+ '/rotro_{}.jpg'.format(count),rostro) #guardamos las capturas en nuestra ruta con su respectivo nombre
        count = count + 1 #incrementamos +1 en el contador
    cv2.imshow('Captura de rostro',frame)

    k =  cv2.waitKey(1) 
    if k == 27 or count >= 300: # si el usuario presiona la tecla ESC (300 en codigo ASSCI) terminara el ciclo de escanar y guardar rostros o si las capturas llegan a 300 tambien
        break

   captura.release()
   cv2.destroyAllWindows()
   os.system("python entrenandoRF.py") #ejecutamos el programa de entrenar la vision artificial

def reconocimiento(): #metodo reconocimiento
    ventana.destroy #destruye la ventana actual
    os.system("python ReconocimientoFacial.py") #abre el programa de reconocimiento facial


ventana = tk.Tk() #inicializamos una ventana usando la libreria de tkiner
ventana.title("AXNAT") #le ponemos un titulo a la ventana
ventana.config(width=400, height=300) #definimos las dimensiones de la ventana
ventana.configure(background='black') #definimos el color de fondo de la ventana
etiqueta_titulo = ttk.Label(text="AXNAT", font="monospaced") #etiqueta que diga AXNAT
#imagen = PhotoImage(file= "candado2.png")
etiqueta_titulo.config(background= "black", foreground= "white") #ponemos atributos de color y fuente a la etiqueta que diga axnat
etiqueta_descripcion = ttk.Label(text="Control de acceso", font="Arial") #etiqueta que dice control de acceso que se llama descripcion
etiqueta_descripcion.config(background="black", foreground= "white") #ponemos atributos a la etiqueta descripcion como color y color de fondo
etiqueta_nombre = ttk.Label(text="Ingresa tu nombre: ") #creamos una etiqueta para indicar que debemos colocar un nombre
etiqueta_nombre.config(background="black", foreground="white") #configuramos la etiqueta de nombre y le ponemos color y color de fondo
etiqueta_nombre.place(x=100, y=90) #damos dimensiones a la etiqueta del nombre
etiqueta_titulo.place(x=170, y=10) #damos dimensiones a la etiqueta de titulo
etiqueta_descripcion.place(x=100, y=40) #le damos dimensiones a la etiqueta de descripcion
caja_usuarios = ttk.Entry() #creamos un textbox para poner nombres llamado caja usuarios
caja_usuarios.place(x=100, y=110, width=200) #damos dimensiones a caja usuarios
caja_usuarios.config(background="#351955", foreground="blue") #ponemos atributos a caja usuarios
boton_agregar = ttk.Button(text="Agregar Alumno", command= persona) #creamos un boton que llama al metodo persona
boton_salir = ttk.Button(text="Iniciar Reconocimiento", command= reconocimiento) #creamos un boton que llama al metodo reconocimiento
boton_salir.place(x=130, y=270) #la damos dimensiones al boton de reconocimiento
boton_agregar.place(x=142, y=140) #le damos dimensiones al boton de agregar usuario
ventana.mainloop() #mantenemos la ventana abierta


