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
path_skew = r'F:\tarun\images\skew\deskew-9.jpg'

image = cv2.imread("blue-red.png")
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
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_copy = otsu.copy()

erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=1)
negated_erode = ~erode_otsu
opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
double_opening_dilated_7x7 = cv2.dilate(double_opening,np.ones((7,7),np.uint8),iterations=2)
double_opening_dilated_7x7_3x3 = cv2.dilate(double_opening_dilated_7x7,np.ones((3,3),np.uint8),iterations=5)


nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(double_opening_dilated_7x7_3x3, connectivity=8)

print(type(nb_components))
print(stats[0][0],stats[0][1])
print("width", stats[0][2])
print("height",stats[0][3])
print(centroids[0])


for i in range(0, nb_components):
    x_min = stats[i, 0]
    y_min = stats[i, 1]
    x_max = stats[i, 0] + stats[i, 2]
    y_max = stats[i, 1] + stats[i, 3]

    cv2.rectangle(sharp_copy2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

contours, heirarchy = cv2.findContours(double_opening_dilated_7x7_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key= cv2.contourArea)

for cnt in range(len(contours)):
    rect = cv2.minAreaRect(contours[cnt])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(sharp_copy,[box],-1,(0,255,0),2)
'''
rect = cv2.minAreaRect(contours[-1])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(sharp_copy,[box],0,(0,0,255),2)
print("angle", rect[2])
'''
#cv2.imwrite("negated_erode.jpg",negated_erode)
cv2.imwrite("sharp_copy.jpg",sharp_copy)
cv2.imwrite("sharp_copy_2.jpg",sharp_copy2)
#cv2.imwrite("erode.jpg",erode_otsu)
#cv2.imwrite("opening.jpg",opening)
#cv2.imwrite("double_opening.jpg",double_opening)
#cv2.imwrite("double_opening_dilated_7x7.jpg",double_opening_dilated_7x7)
cv2.imwrite("double_opening_dilated_7x7_3x3.jpg",double_opening_dilated_7x7_3x3)
#cv2.imwrite("opening.jpg",opening)

#cv2.imshow("resized",image_resized)
#cv2.imshow("deskewed",deskewed)
#cv2.imshow("text",sharp_copy)
#cv2.imshow("gaussia",gaussian)
#cv2.imshow("equalised",equalised)
#cv2.imshow("gray",gray)
#cv2.imshow("bitwise_not",bitwise_not)
#cv2.imshow("sharp",sharp)
#cv2.imshow("otsu1",image_resized1)
#cv2.imshow("dilate",dilate)
#cv2.imshow("opening",opening)
cv2.imshow("sharp_copy",sharp_copy)
cv2.imshow("negated_erode",negated_erode)



#cv2.imshow("closing",closing)

cv2.waitKey(0)
cv2.destroyAllWindows()