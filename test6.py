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
path_skew = r'F:\tarun\images\skew\deskew-9.jpg'

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
gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
#gray = ~gray
gray_copy = gray.copy()
_, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
'''
erode_otsu = cv2.erode(t2,np.ones((7,7),np.uint8),iterations=2)
negated_erode = ~erode_otsu    
opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((1,5),np.uint8),iterations=6)
opening2 = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((5,1),np.uint8),iterations=6)
double_opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
double_opening_dilated_3x3 = cv2.dilate(double_opening,np.ones((1,2),np.uint8),iterations=3)
contours_dilation,hierarchy = cv2.findContours(negated_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
'''

erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=1)
negated_erode = ~erode_otsu
opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
opening2 = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((1,2),np.uint8),iterations=9)
opening3 = cv2.morphologyEx(opening2,cv2.MORPH_OPEN,np.ones((2,1),np.uint8),iterations=9)
double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
double_opening_dilated_3x3 = cv2.dilate(double_opening,np.ones((3,3),np.uint8),iterations=4)
contours_dilation,hierarchy = cv2.findContours(double_opening_dilated_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

edges = cv2.Canny(otsu,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,100)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1600*(-b))
    y1 = int(y0 + 1600*(a))
    x2 = int(x0 - 1600*(-b))
    y2 = int(y0 - 1600*(a))

    cv2.line(deskewed1,(x1,y1),(x2,y2),(0,0,255),4)


#areas = []
widths = []
heights = []

for cnt in range(len(contours_dilation)):
    x,y,w,h = cv2.boundingRect(contours_dilation[cnt])
    #areas.append(getcontourarea(contours_dilation[cnt]))
    heights.append(h)
    widths.append(w)

mean_heights = np.mean(heights)
mean_widths = np.mean(widths)
#mean_area = np.mean(areas)
#print(mean_widths,mean_heights,mean_area)

for cnt in range(len(contours_dilation)):
    x,y,w,h = cv2.boundingRect(contours_dilation[cnt])
    if ((w > (mean_widths - 25))):
        if(h > (mean_heights - 25)):
            cv2.rectangle(sharp_copy1,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imwrite("otsu.jpg",otsu)
cv2.imwrite("negated_erode.jpg",negated_erode)
cv2.imwrite("opening2.jpg",opening2)
cv2.imwrite("opening3.jpg",opening3)
cv2.imwrite("opening.jpg",opening)
cv2.imwrite("double opening.jpg",double_opening)
cv2.imwrite("double opening dialted 3x3.jpg",double_opening_dilated_3x3)
cv2.imwrite("sharp copy1.jpg",sharp_copy1)
cv2.imwrite('houghlines3.jpg',deskewed1)
cv2.imwrite('edges.jpg',edges)
#cv2.imwrite('deskewed.jpg',deskewed)