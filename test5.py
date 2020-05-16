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
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
image_resized = IMAGE_TOOLS_LIB.image_resize(no_shadows,1600,1200)
gaussian_blur = cv2.GaussianBlur(image_resized,(5,5),0)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
sharp_copy = sharp.copy()
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
t2 = v
for p in range(image_resized.shape[0]):
    for q in range(image_resized.shape[1]):
        pixel = t2[p,q]
        if (pixel > val2):
            t2[p,q] = 255
        else:
            t2[p,q] = 0
        
t2_copy = t2.copy()


custom_oem_psm_config1 = r'--oem 3 --psm 12'
ocr = pytesseract.image_to_data(t2, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
print(len(ocr['text']))

boxes = len(ocr['text'])
engine = pyttsx3.init()
for i in range(boxes):
    if int(ocr['conf'][i])>60:
        print(ocr['text'][i])
        (x,y,w,h) = (ocr['left'][i],ocr['top'][i],ocr['width'][i],ocr['height'][i])
        cv2.rectangle(t2_copy,(x,y),(x+w,y+h),(0,0,255),1)
        #cv2.imshow("text",t2_copy)
        #engine.say(ocr['text'][i])
        #engine.runAndWait()
        #cv2.waitKey(10)


#cv2.imshow("t2",t2)
cv2.imwrite("t2_with_sharp_5x5.jpg",t2)
#cv2.imwrite("t2_with_sharp_5x5_text_val2.jpg",t2_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()






