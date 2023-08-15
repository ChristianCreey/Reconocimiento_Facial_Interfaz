import cv2
import os
import imutils
#crear carpeta para guardar imagenes
modelo = 'Fotosyo'
ruta1 = 'C:/Users/Christian Creey/Documents/PycharmProjects/untitled/RecFaPython/reconocimientoFacial1'
rutacompleta = ruta1 + '/' + modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)#makedirs permite crear una carpeta

ruidos = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#encedemos camara
camara = cv2.VideoCapture(0)
id = 350
while True:
    respuesta, capture = camara.read()
    if respuesta==False:
        break
    captura = imutils.resize(capture, width=640)
    #pasamos a grises
    grises = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    idcaptura = capture.copy()
    caras = ruidos.detectMultiScale(grises,1.1,minNeighbors=5)
    for(x,y,e1,e2) in caras:
        cv2.rectangle(capture,(x,y), (x+e1, y+e2), (255,0,0), 2)
        rostrocapturado = idcaptura[y:y+e2,x:x+e1]#cordenada de las capturas
        rostrocapturado = cv2.resize(rostrocapturado, (160,160), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado)#modificar el nomre de las imagenes
        id += 1

    cv2.imshow("RESULTADO", capture)

    if id == 500:
        break
camara.release()
cv2.destroyAllWindows()
