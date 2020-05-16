import numpy as np 
import cv2 
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
path_skew = r'F:\tarun\images\skew\deskew-19.jpg'

image = cv2.imread("3.jpeg")
deskewed = DESKEW_MAR.correct_skew(image)
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
image_resized = IMAGE_TOOLS_LIB.image_resize(no_shadows,1600,1200)
gaussian_blur = cv2.GaussianBlur(image_resized,(5,5),0)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
#gray = ~gray
gray_copy = gray.copy()
sharp_copy = sharp.copy()

#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
#equalised = clahe.apply(gray)

#gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#smooth = cv2.addWeighted(gaussian,1.5,deskewed,-0.5,0)
#bilateral = cv2.bilateralFilter(deskewed,9,75,75)

binary_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
binary_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
coords_black = np.column_stack(np.where(gray < (ret + 30)))
coords_white = np.column_stack(np.where(gray > (ret + 30)))
print(len(coords_black),len(coords_white))

for i in range(len(coords_black)):
    gray_copy[coords_black[i][0],coords_black[i][1]] = 0
    
for i in range(len(coords_white)):
    gray_copy[coords_white[i][0],coords_white[i][1]] = 255


custom_oem_psm_config1 = r'--oem 3 --psm 12'
#custom_oem_psm_config2 = r'--oem 1 --psm 12'
#custom_oem_psm_config2 = r'--oem 3 --psm 11'
res = cv2.imread("res1.jpeg")
#ocr1 = pytesseract.image_to_data(gray1, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
ocr = pytesseract.image_to_data(res, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#ocr3= pytesseract.image_to_data(gray3, output_type=Output.DICT,config=custom_oem_psm_config2,lang='eng')
#print(d.keys())
print(len(ocr['text']))

boxes = len(ocr['text'])
#engine = pyttsx3.init()
for i in range(boxes):
    if int(ocr['conf'][i])>60:
        #print(d2['text'][i])
        (x,y,w,h) = (ocr['left'][i],ocr['top'][i],ocr['width'][i],ocr['height'][i])
        cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255),1)
        #cv2.imshow("text",erode)
        #engine.say(d2['text'][i])
        #engine.runAndWait()
        #cv2.waitKey(10)
#cv2.imshow("sharp copy",sharp_copy)
cv2.imwrite("sharp copy.jpg",sharp_copy)
cv2.imwrite("gray.jpg",gray)
cv2.imwrite("gray_copy.jpg",gray_copy)
cv2.imwrite("binary_gaussian.jpg",binary_gaussian)
cv2.imwrite("binary_mean.jpg",binary_mean)
cv2.imwrite("otsu.jpg",otsu)
cv2.imwrite("res.jpg",res)
cv2.waitKey(0)
cv2.destroyAllWindows()