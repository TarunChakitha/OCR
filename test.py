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

image = cv2.imread(path_skew)
image_resized = IMAGE_TOOLS_LIB.image_resize(image,2000,3000)
image_resized1 = image_resized.copy()
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image_resized)
gaussian_blur = cv2.GaussianBlur(no_shadows,(5,5),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
gray_copy = gray.copy()
sharp_copy = sharp.copy()
sharp_copy2 = sharp.copy()
adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_copy = otsu.copy()

skew = determine_skew(gray)

erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=1)
negated_erode = ~erode_otsu
opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
double_opening_dilated_7x7 = cv2.dilate(double_opening,np.ones((3,3),np.uint8),iterations=4)
double_opening_dilated_7x7_3x3 = cv2.dilate(double_opening_dilated_7x7,np.ones((3,3),np.uint8),iterations=10)
double_opening_dilated_7x7_3x3_copy = double_opening_dilated_7x7_3x3.copy()
double_opening_dilated_7x7_copy = double_opening_dilated_7x7.copy()
contours_otsu,hierarchy = cv2.findContours(double_opening_dilated_7x7,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
act_angles = []
for cnt in range(len(contours_otsu)):        
    cnt=cv2.convexHull(contours_otsu[cnt])
    angle = cv2.minAreaRect(cnt)[-1]
    #print("Actual angle is:"+str(angle))
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(double_opening_dilated_7x7_copy,[box],0,(0,0,255),1)
    p = np.array(rect[1])
    if (p[0] < p[1]):
        act_angles.append(rect[-1]+180)
    else:
        act_angles.append(rect[-1]+90)
'''
p=np.array(rect[1])
if p[0] < p[1]:
        print("Angle along the longer side:"+str(rect[-1] + 180))
        act_angle=rect[-1]+180
else:
        print("Angle along the longer side:"+str(rect[-1] + 90))
        act_angle=rect[-1]+90
#act_angle gives the angle of the minAreaRect with the vertical
'''
mean_angles = np.mean(act_angles)
median_angles = np.median(act_angles)
print(mean_angles,median_angles)

if median_angles < 90:
        median_angles = (90 + median_angles)
        print("angleless than -45")
 
        # otherwise, just take the inverse of the angle to make
        # it positive
else:
        median_angles=median_angles-180
        print("grter than 90")
print(median_angles)


if mean_angles < 90:
        mean_angles = (90 + mean_angles)
        print("angleless than -45")
 
        # otherwise, just take the inverse of the angle to make
        # it positive
else:
        mean_angles=mean_angles-180
        print("grter than 90")
print(mean_angles)



# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, median_angles, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



cv2.imwrite("MinAreaRect9.jpg",double_opening_dilated_7x7_copy)
cv2.imwrite('Deskewed_9.jpg', rotated)
