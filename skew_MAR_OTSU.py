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
#gaussian_blur = cv2.GaussianBlur(deskewed,(7,7),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
gray_copy = gray.copy()
sharp_copy = sharp.copy()
sharp_copy2 = sharp.copy()
#bitwise_not = cv2.bitwise_not(gray)
adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
#binary_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#print(ret)
#binary_gaussian1 = binary_gaussian.copy()
#binary_mean1 = binary_mean.copy()
otsu_copy = otsu.copy()





#binary_images = [binary_gaussian1,binary_mean1,otsu1]
#dilate = cv2.dilate(binary_gaussian,np.ones((3,3),np.uint8))
#erode_gaussian = cv2.erode(binary_gaussian,np.ones((7,7),np.uint8))
#erode_mean = cv2.erode(binary_mean,np.ones((7,7),np.uint8))
erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=3)
erode_otsu_copy = erode_otsu.copy()
negated_erode = ~erode_otsu
negated_erode_copy = negated_erode.copy()
#closing = cv2.morphologyEx(binary_gaussian,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

#contours_gaussian,hierarchy = cv2.findContours(~erode_gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours_mean,hierarchy = cv2.findContours(~erode_mean,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_otsu,hierarchy = cv2.findContours(negated_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours_otsu),type(contours_otsu))
unwanted_contours = []
#contours = [contours_gaussian,contours_mean,contours_otsu]
#print(len(contours_gaussian),len(contours_mean),len(contours_otsu))
#angles_gaussian = []
#angles_mean = []
contour_areas = []

#angles = [angles_gaussian,angles_mean,angles_otsu]
for cnt in range(len(contours_otsu)):
    contour_areas.append(cv2.contourArea(contours_otsu[cnt]))
    
contour_areas.sort()

mean_area = np.mean(contour_areas)
median_area = np.median(contour_areas)

angles = []
heights = []
widths = []

for cnt in range(len(contours_otsu)):
    if ((cv2.contourArea(contours_otsu[cnt]) < mean_area + 500)):
        rect = cv2.minAreaRect(contours_otsu[cnt])
        (y1,x1), (y2,x2), angle = rect
        height = abs(x1 - x2)
        width = abs(y1 - y2)
        heights.append(height)
        widths.append(width)
        angles.append(angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(negated_erode_copy,[box],-1,(0,0,0),-1)
        
# ((cv2.contourArea(contours_otsu[cnt]) < mean + 300) and (cv2.contourArea(contours_otsu[cnt]) > mean - 200))
negated_erode_copy = cv2.morphologyEx(negated_erode_copy,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(negated_erode_copy, connectivity=8)

for i in range(0, nb_components):
    x_min = stats[i, 0]
    y_min = stats[i, 1]
    x_max = stats[i, 0] + stats[i, 2]
    y_max = stats[i, 1] + stats[i, 3]

    cv2.rectangle(sharp_copy2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)




mean_heights = np.mean(heights)
mean_widths = np.mean(widths)
mean_angles = np.mean(angles)

print("mean heights", mean_heights)
print("mean widths", mean_widths)
print("mean angles", mean_angles)

if (mean_heights > mean_widths):
    connecting_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
else:
    connecting_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,15))

connected = cv2.dilate(negated_erode_copy,connecting_kernel,iterations= 2)



final_contours, x = cv2.findContours(connected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(final_contours,key= cv2.contourArea)

cnt1 = sorted_contours[-1]
rect = cv2.minAreaRect(cnt1)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(sharp_copy,[box],0,(0,255,0),5)

cnt2 = sorted_contours[-2]
rect = cv2.minAreaRect(cnt2)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(sharp_copy,[box],0,(0,255,0),5)

cnt3 = sorted_contours[-3]
rect = cv2.minAreaRect(cnt3)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(sharp_copy,[box],0,(0,255,0),5)

x,y,w,h = cv2.boundingRect(cnt1)
cv2.rectangle(sharp_copy,(x,y),(x+w,y+h),(255,0,0),5)

x,y,w,h = cv2.boundingRect(cnt2)
cv2.rectangle(sharp_copy,(x,y),(x+w,y+h),(255,0,0),5)

x,y,w,h = cv2.boundingRect(cnt3)
cv2.rectangle(sharp_copy,(x,y),(x+w,y+h),(255,0,0),5)


cv2.imwrite("negated_erode.jpg",negated_erode)
cv2.imwrite("negated_erode_copy.jpg",negated_erode_copy)
cv2.imwrite("connected.jpg",connected)
cv2.imwrite("sharp_copy.jpg",sharp_copy)
cv2.imwrite("sharp_copy_2.jpg",sharp_copy2)
cv2.imwrite("erode_iterated.jpg",erode_otsu)
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
cv2.imshow("negated_erode_copy",negated_erode_copy)
cv2.imshow("connected",connected)

#cv2.imshow("closing",closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
