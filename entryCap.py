import cv2 as cv
import os
import imutils


model = "\oscar"
route1 = r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\data"
completeRoute = route1+model

if not os.path.exists(completeRoute):
    os.makedirs(completeRoute)

# xml de openCv con la ia para la cara de la persona
ruidos=cv.CascadeClassifier(r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml")
camera = cv.VideoCapture(0)

imageId = 350
while True:
    response,frame = camera.read()
    if response == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    captureId = frame.copy()
    face=ruidos.detectMultiScale(gray, 1.3, 5)
    for (x,y,e1,e2) in face:
        cv.rectangle(frame, (x,y), (x+e1,y+e2), (0, 255, 0), 2)
        faceCapture=captureId[y:y+e2, x:x+e1]
        faceCapture=cv.resize(faceCapture, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(completeRoute+'/image_{}.jpg'.format(imageId), faceCapture)
        imageId += 1
    cv.imshow("reconocimiento", frame)
    print(imageId)
    if imageId == 600:
        break

camera.release()
cv.destroyAllWindows()