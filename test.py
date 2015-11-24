# -*- coding: utf-8 -*-

import numpy as np
import cv2, slda


# Number of topics to use for LDA
nTopics = 8

# Load all image files
data = np.zeros((1000,16,2),dtype=np.uint8)
for i in range(1000):
    data[i,:,0] = np.asarray(range(16)).astype(np.int)
    data[i,:,1] = cv2.imread("image/%d.bmp"%i,0).reshape((16)).astype(np.uint8)
label = np.loadtxt("image/label.csv",delimiter=",")

sLDA = slda.sLDA()
sLDA.setData(data,label)
sLDA.solve(epsilon=1e-4)

# Show topics obtained
for i in range(8):
    out = np.zeros((4,4))
    for j in range(255*4):
        out += np.random.dirichlet(sLDA.qPhi[i,:]).reshape((4,4))
    cv2.imwrite("%d_%.1f.bmp"%(i,sLDA.eta[i]),cv2.resize(out.astype(np.uint8),(200,200),interpolation=cv2.INTER_NEAREST))

print sLDA.eta