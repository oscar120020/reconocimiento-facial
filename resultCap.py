import cv2 as cv
import os

data = r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\data"
dataList = os.listdir(data)

trainingModel = cv.face.EigenFaceRecognizer_create()
trainingModel.read(r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\TrainingEigenFaceRecognizer.xml")
ruidos = cv.CascadeClassifier(r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml")

Camera = cv.VideoCapture(0)

while True:
    response, frame = Camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    captureId = gray.copy()
    face = ruidos.detectMultiScale(gray, 1.3, 7)
    for (x,y,e1,e2) in face:        
        faceCapture=captureId[y:y+e2, x:x+e1]
        faceCapture=cv.resize(faceCapture, (160, 160), interpolation=cv.INTER_CUBIC)
        result = trainingModel.predict(faceCapture)
        cv.putText(frame, '{}'.format(result), (x, y-5), 1, 1.2, (0, 255, 0), 2, cv.LINE_AA)
        if result[1] < 10000:
            cv.putText(frame, '{}'.format(dataList[result[0]]), (x, y-20), 1, 1.3, (0, 255, 0), 2, cv.LINE_AA)            
            cv.rectangle(frame, (x,y), (x+e1, y+e2), (255, 0, 0), 2)
        else:
            cv.putText(frame, ('No encontrado'), (x, y-20), 2, 1.1, (0, 255, 0), 2, cv.LINE_AA)            
            cv.rectangle(frame, (x,y), (x+e1, y+e2), (255, 0, 0), 2)
    cv.imshow("reconocimineto", frame)
    if(cv.waitKey(1) == ord("q")):
        break
Camera.release()
cv.destroyAllWindows()