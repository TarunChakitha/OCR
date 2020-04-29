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
image_resized = IMAGE_TOOLS_LIB.image_resize(image,1200,675)
image_resized1 = image_resized.copy()
image_resized2 = image_resized.copy()
image_resized3 = image_resized.copy()
images = [image_resized1,image_resized2,image_resized3]
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image_resized)
#deskewed ,angle = rotate(image_resized,(255,255,255))
gaussian_blur = cv2.GaussianBlur(no_shadows,(5,5),0)
#gaussian_blur = cv2.GaussianBlur(deskewed,(7,7),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
sharp_copy = sharp.copy()
#bitwise_not = cv2.bitwise_not(gray)
binary_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
binary_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
binary_gaussian1 = binary_gaussian.copy()
binary_mean1 = binary_mean.copy()
otsu1 = otsu.copy()

binary_images = [binary_gaussian1,binary_mean1,otsu1]
#dilate = cv2.dilate(binary_gaussian,np.ones((3,3),np.uint8))
erode_gaussian = cv2.erode(binary_gaussian,np.ones((7,7),np.uint8))
erode_mean = cv2.erode(binary_mean,np.ones((7,7),np.uint8))
erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8))
#closing = cv2.morphologyEx(binary_gaussian,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

contours_gaussian,hierarchy = cv2.findContours(~erode_gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_mean,hierarchy = cv2.findContours(~erode_mean,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_otsu,hierarchy = cv2.findContours(~erode_otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = [contours_gaussian,contours_mean,contours_otsu]
print(len(contours_gaussian),len(contours_mean),len(contours_otsu))
angles_gaussian = []
angles_mean = []
angles_otsu = []
angles = [angles_gaussian,angles_mean,angles_otsu]
for j in range(len(contours)):
    for cnt in range(len(contours[j])):
        if ((cv2.contourArea(contours[j][cnt]) > 1000)):
            rect = cv2.minAreaRect(contours[j][cnt])
            if (rect[2] != -0):
                angles[j].append(rect[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                for i in range(len(images)):
                    cv2.drawContours(images[i],[box],-1,(255,0,0),1)
'''
for cnt in range(len(contours_mean)):
    rect = cv2.minAreaRect(contours_mean[cnt])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(binary_images[1],[box],-1,(255,0,0),2)

for cnt in range(len(contours_otsu)):
    rect = cv2.minAreaRect(contours_otsu[cnt])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(binary_images[2],[box],-1,(255,0,0),2)

'''
for i in range(len(angles)):
    angles[i] = angles[i].sort()



print("angles gaussian =" ,angles_gaussian)
print("angles mean =" ,angles_mean)
print("angles otsu =" ,angles_otsu)

mean_gaussian = np.mean(angles_gaussian)
mean_mean = np.mean(angles_mean)
mean_otsu = np.mean(angles_otsu)

median_gaussian = np.median(angles_gaussian)
median_mean = np.median(angles_mean)
median_otsu = np.median(angles_otsu)

print(mean_gaussian,mean_mean,mean_otsu)
print(median_gaussian,median_mean,median_otsu)

#cv2.imshow("resized",image_resized)
#cv2.imshow("deskewed",deskewed)
#cv2.imshow("text",sharp_copy)
#cv2.imshow("gaussia",gaussian)
#cv2.imshow("equalised",equalised)
#cv2.imshow("gray",gray)
#cv2.imshow("bitwise_not",bitwise_not)
#cv2.imshow("sharp",sharp)

cv2.imshow("binary_mean1",image_resized2)
cv2.imshow("binary_gaussian1",image_resized1)
cv2.imshow("otsu1",image_resized3)
#cv2.imshow("dilate",dilate)
cv2.imshow("erode_gaussian",~erode_gaussian)
cv2.imshow("erode_mean",~erode_mean)
cv2.imshow("erode_otsu",~erode_otsu)
#cv2.imshow("closing",closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
