import numpy as np 
import cv2 
import math
import pyttsx3
import pytesseract
import IMAGE_TOOLS_LIB
from deskew import determine_skew, determine_skew_dev
from typing import Tuple, Union
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

img = cv2.imread("text1.png")
#copy of original image
img1 = img.copy()
img2 = img.copy()

# original image
#cv2.imshow("original",img)

# median blur
median_blur = cv2.medianBlur(img,5)
#cv2.imshow("median_blur",median_blur)

# gray scale image
gray= cv2.cvtColor(median_blur,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",gray)

# otsu
ret,otsu = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
#ret,otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
#cv2.imshow("otsu",otsu)
#otsu1 = otsu.copy()

# applying bitwise not to invert color
inverted = cv2.bitwise_not(otsu)
#cv2.imshow("inverted",inverted)

# dilation
dilate = cv2.dilate(inverted,np.ones((3,3),np.uint8),iterations = 2)
#cv2.namedWindow("dilate",cv2.WINDOW_NORMAL)
#cv2.imshow("dilate",dilate)

#img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#mask = img_gray & dilate
#cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
#cv2.imshow("mask",mask)

# find contours
contours, hierarcky = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# min area rect and boxpoints 
x,y,h,w = cv2.boundingRect(max(contours,key = cv2.contourArea))
print(x,y,h,w)
#box = cv2.boxPoints(rect)
#print(box)
#box = np.int0(box)
rectangle = cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)
cv2.imshow("rectangle",rectangle)

mask = np.zeros(img.shape,np.uint8)
mask[y:y+w,x:x+h] = 255
cv2.imshow("mask",mask)
masking = mask & img
#cv2.imwrite("masked_stop.jpg",masking)
cv2.imshow("masking",masking)

cv2.waitKey(0)
cv2.destroyAllWindows()
