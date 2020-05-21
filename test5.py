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
path_shadows = r'F:\tarun\images\shadows\shadows12.jpg'
path_skew = r'F:\tarun\images\skew\deskew-19.jpg'

image = cv2.imread(path_shadows)
deskewed = DESKEW_MAR.correct_skew(image)
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
_, otsu = cv2.threshold(v,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=1)
negated_erode = ~erode_otsu    
opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
double_opening_dilated_3x3 = cv2.dilate(double_opening,np.ones((3,3),np.uint8),iterations=4)
contours_dilation,hierarchy = cv2.findContours(negated_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

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
print("dilation width and hieght",mean_widths,mean_heights)

for cnt in range(len(contours_dilation)):
    x,y,w,h = cv2.boundingRect(contours_dilation[cnt])
    if ((w > (mean_widths - 10))):
        if(h > (mean_heights - 10)):
            if(h > (mean_heights - 10)):
                        cv2.rectangle(sharp_copy1,(x,y),(x+w,y+h),(255,0,0),2)

def getcontourarea(cnt):
    return cv2.contourArea(cnt)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,1))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,4))
connected = cv2.erode(otsu,kernel,iterations=3)
connected = cv2.erode(connected,kernel2,iterations=1)
connected = ~connected
contours, _ = cv2.findContours(connected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#areas = []
widths = []
heights = []
for cnt in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[cnt])
    #areas.append(getcontourarea(contours[cnt]))
    heights.append(h)
    widths.append(w)

mean_heights = np.mean(heights)
mean_widths = np.mean(widths)
#mean_area = np.mean(areas)
print("connected width and hieght",mean_widths,mean_heights)

for cnt in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[cnt])
    if (w > (mean_widths - 10)):
        if(h > (mean_heights - 10)):
            if (h < (mean_heights + 50)):
                cv2.rectangle(sharp_copy2,(x,y),(x+w,y+h),(255,0,0),2)

#custom_oem_psm_config1 = r'--oem 3 --psm 12'
#ocr = pytesseract.image_to_data(t2, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#print(ocr)
#print(len(ocr['text']))

cv2.imwrite("connected.jpg",connected)
cv2.imwrite("negated erode.jpg",negated_erode)
cv2.imwrite("sharp copy1.jpg",sharp_copy1)
cv2.imwrite("sharp copy2.jpg",sharp_copy2)
#cv2.imwrite("sharp copy3.jpg",sharp_copy3)
cv2.waitKey(0)
cv2.destroyAllWindows()






