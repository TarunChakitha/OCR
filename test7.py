import cv2
import numpy as np
import math
import pyttsx3
import pytesseract
import IMAGE_TOOLS_LIB
import re
import DESKEW_MAR
from deskew import determine_skew
from typing import Tuple, Union
from pytesseract import Output
from matplotlib import pyplot as plt


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

image = cv2.imread(path_skew)
deskewed = DESKEW_MAR.correct_skew(image)
deskewed1 = deskewed.copy()
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
image_resized = IMAGE_TOOLS_LIB.image_resize(no_shadows,1600,1200)

gaussian_blur = cv2.GaussianBlur(image_resized,(5,5),0)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
sharp_copy1 = sharp.copy()
sharp_copy2 = sharp.copy()
sharp_copy3 = sharp.copy()
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
#gray = ~gray
gray_copy = gray.copy()
hsv = cv2.cvtColor(sharp,cv2.COLOR_BGR2HSV)
v = hsv[:,:,2]
m = np.mean(v[:])
s = np.std(v[:])
k = -0.4
value = m + k*s
#temp = v

# sauvola
val2 = m*(1+0.1*((s/128)-1))
print(value,val2)
t2 = v
for p in range(image_resized.shape[0]):
    for q in range(image_resized.shape[1]):
        pixel = t2[p,q]
        if (pixel > value):
            t2[p,q] = 255
        else:
            t2[p,q] = 0
        
t2_copy = t2.copy()
t2_copy2 = t2.copy()
_, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#It = cv2.bitwise_not(otsu)
_,labels = cv2.connectedComponents(t2)

result = np.zeros((gray.shape[0],gray.shape[1],3),np.uint8)

for i in range(labels.min(),labels.max()+1):
    mask = cv2.compare(labels,i,cv2.CMP_EQ)

    ctrs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    result = cv2.drawContours(t2_copy,ctrs,-1,(255,255,255))
cv2.imwrite("t2.jpg",t2)
cv2.imwrite("t2 copy.jpg",t2_copy)