# -*- coding: utf-8 -*-

import numpy as np
import cv2,os


topic1 = np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],dtype=np.float32)*255
topic2 = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],dtype=np.float32)*255
topic3 = np.array([0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],dtype=np.float32)*255
topic4 = np.array([0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],dtype=np.float32)*255

topic5 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)*255
topic6 = np.array([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],dtype=np.float32)*255
topic7 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],dtype=np.float32)*255
topic8 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],dtype=np.float32)*255

labels = []

if not os.path.exists("image"):
    os.mkdir("image")

for i in range(1000):

    alpha = np.full(8,1,dtype=np.float32)
    theta = np.random.dirichlet(alpha)
    outcome = np.zeros(16,dtype=np.float32)
    label = 0.0

    outcome += theta[0]*topic1
    outcome += theta[1]*topic2
    outcome += theta[2]*topic3
    outcome += theta[3]*topic4
    outcome += theta[4]*topic5
    outcome += theta[5]*topic6
    outcome += theta[6]*topic7
    outcome += theta[7]*topic8

    label += theta[0]*-1.0
    label += theta[1]*-0.8
    label += theta[2]*-0.5
    label += theta[3]*-0.2
    label += theta[4]*0.1
    label += theta[5]*0.4
    label += theta[6]*0.7
    label += theta[7]*1.0

    labels.append( ((label+1.0)/2.0)>=0.5 )

    cv2.imwrite("image/%d.bmp"%i,outcome.reshape((4,4)).astype(np.uint8))

print "Number of Images with label==1 : %d"%np.sum(labels)
np.savetxt("image/label.csv",labels,delimiter=",")



