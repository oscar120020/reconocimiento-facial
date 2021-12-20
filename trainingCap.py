import cv2 as cv
import os
import numpy as np
from time import time
data = r"C:\Users\intellisys\Desktop\pyhton\reconocimiento\data"
dataList = os.listdir(data)

ids=[]
dataFace = []
id=0
initialTime = time()
for row in dataList:
    completeRoute = data + "\{}".format(row)
    for file in os.listdir(completeRoute):
        print(row + "/" + file)
        ids.append(id)
        dataFace.append(cv.imread("{}\{}".format(completeRoute, file), 0))
    id += 1

finalTime = time()
training = cv.face.EigenFaceRecognizer_create()
training.train(dataFace, np.array(ids))
print("Start trining...wait")
print("timepo total", finalTime - initialTime)

training.write("TrainingEigenFaceRecognizer.xml")
print("Training done")