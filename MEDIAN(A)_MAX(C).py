import numpy as np 
import cv2 
import math
import pyttsx3
import pytesseract
import IMAGE_TOOLS_LIB
import re
import struct
import zlib
from deskew import determine_skew, determine_skew_dev
from typing import Tuple, Union
from pytesseract import Output
from matplotlib import pyplot as plt
from PIL import Image
import struct
import numpy as np
import cv2
import zlib



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-1.png'

image = cv2.imread(path_skew)
#RGBimage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#PILimage = Image.fromarray(RGBimage)
#PILimage.save("result.png",dpi = (300,300))
#image = cv2.imread("result.png")
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
contours_otsu,hierarchy = cv2.findContours(double_opening_dilated_7x7,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contour_areas = []

for cnt in range(len(contours_otsu)):
    contour_areas.append(cv2.contourArea(contours_otsu[cnt]))
    
contour_areas.sort()

mean_area = np.mean(contour_areas)
median_area = np.median(contour_areas)

#print(mean_area, median_area)

angles = []

def getcontourarea(cnt):
    return cv2.contourArea(cnt)

for cnt in range(len(contours_otsu)):
    rect = cv2.minAreaRect(contours_otsu[cnt])
    angles.append(rect[-1])
    
angles.sort()
#print(angles)
mean_angles = np.mean(angles)
median_angles = np.median(angles)
mode_angles = (3*median_angles - 2*mean_angles)


print(mean_angles,median_angles,mode_angles)

def complexAngle(angle):
    if 0 <= angle <= 90:
        corrected_angle = angle - 90
    elif -45 <= angle < 0:
        corrected_angle = angle - 90
    elif -90 <= angle < -45:
        corrected_angle = 90 + angle
    return corrected_angle

def simpleAngle(angle):
    if median_angles <= -45:
        angle_simple = (90 + median_angles)
    elif median_angles >= +45:
        angle_simple = (-90 + median_angles)
    else:
        angle_simple = median_angles
    return angle_simple

print("corrected deskew library",skew)
print("corrected mar median angle complex", complexAngle(median_angles))
print("corrected mar angle simle",simpleAngle(median_angles))

def rotate(image: np.ndarray,angle, background_color): # OFFIAL DOCUMENTATION
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background_color)


rotated_deskew = rotate(image_resized,skew,(0,0,0))
rotated_median_simple = rotate(image_resized,simpleAngle(median_angles),(0,0,0))
rotated_median_complex = rotate(image_resized,complexAngle(median_angles),(0,0,0))

contours , hierarchy = cv2.findContours(double_opening_dilated_7x7_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea)
max_contour = contours[-1]
rect = cv2.minAreaRect(max_contour)
max_contour_angle = rect[-1]
box = cv2.boxPoints(rect)
box = np.int0(box)
#print("box")
#print("\n")
#print(box)
cv2.drawContours(image_resized1,[box],0,(0,255,0),2)
print("max contour angle", max_contour_angle)
print("corrected max contour angle complex", complexAngle(max_contour_angle))
print("corrected max contour angle simple", simpleAngle(max_contour_angle))

max_contour_rotated_complex = rotate(image_resized1,complexAngle(max_contour_angle),(0,0,0))
max_contour_rotated_simple = rotate(image_resized1,simpleAngle(max_contour_angle),(0,0,0))

rotated_median_complex_gray = cv2.cvtColor(rotated_median_complex,cv2.COLOR_BGR2GRAY)
rotated_median_complex_gaussian = cv2.adaptiveThreshold(rotated_median_complex_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
#rotated_median_complex_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
#ret, rotated_median_complex_otsu = cv2.threshold(rotated_median_complex_gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("ada gau.jpg",rotated_median_complex_gaussian)

#rotated_median_complex = setdpi(rotated_median_complex)
#rotated_median_complex = np.array(rotated_median_complex,dtype=np.uint8)
#osd_original = pytesseract.image_to_osd(image_resized)
#osd_rotated_deskew = pytesseract.image_to_osd(rotated_deskew)
osd_rotated_median_complex = pytesseract.image_to_osd(rotated_median_complex_gaussian)
#angle_original = re.search('(?<=Rotate: )\d+', osd_original).group(0)
#angle_rotated_deskew = re.search('(?<=Rotate: )\d+', osd_rotated_deskew).group(0)
angle_rotated_median_complex = re.search('(?<=Rotate: )\d+', osd_rotated_median_complex).group(0)
#print("angle: original", angle_original)
print("angle: rotated median complex", angle_rotated_median_complex)
#print("angle: rotated deskew", angle_rotated_deskew)
print(int(angle_rotated_median_complex))

if (angle_rotated_median_complex == '0'):
    print("angle rotated median complex 0")
    print("no second rotation")
    cv2.imshow("rotated median complex",rotated_median_complex)
elif (angle_rotated_median_complex == '90'):
    print("osd second rotation angle 90")
    second_rotation = rotate(rotated_median_complex,90,(0,0,0))
    cv2.imshow("second_rotation",second_rotation)
    cv2.imwrite("second rotation.jpg",second_rotation)
elif (angle_rotated_median_complex == '180'):
    print("osd second rotation angle 180")
    second_rotation = rotate(rotated_median_complex,180,(0,0,0))
    cv2.imshow("second_rotation",second_rotation)
    cv2.imwrite("second rotation.jpg",second_rotation)
elif (angle_rotated_median_complex == '270'):
    print("osd second rotation angle 270")
    second_rotation = rotate(rotated_median_complex,90,(0,0,0))
    cv2.imshow("second_rotation",second_rotation)
    cv2.imwrite("second rotation.jpg",second_rotation)

titles = ['original','rotated_deskew', 'rotated median complex',
            'max contour complex','max contour simple','rotated median simple','rot med gau','rot med gray']
images = [image, rotated_deskew, rotated_median_complex, max_contour_rotated_complex,
            max_contour_rotated_simple,rotated_median_simple,rotated_median_complex_gaussian,rotated_median_complex_gray]

for i in range(8):
    plt.subplot(4,2,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    #plt.xticks([]),plt.yticks([])
    plt.axis("off")
plt.show()


#cv2.imwrite("negated_erode.jpg",negated_erode)
#cv2.imwrite("sharp_copy.jpg",sharp_copy)
#cv2.imwrite("sharp_copy_2.jpg",sharp_copy2)
#cv2.imwrite("erode.jpg",erode_otsu)
#cv2.imwrite("opening.jpg",opening)
#cv2.imwrite("double_opening.jpg",double_opening)
#cv2.imwrite("double_opening_dilated_7x7.jpg",double_opening_dilated_7x7)
#cv2.imwrite("double_opening_dilated_7x7_3x3.jpg",double_opening_dilated_7x7_3x3)
#cv2.imwrite("double_opening_dilated_7x7_copy.jpg",double_opening_dilated_7x7_copy)
#cv2.imwrite("rotated_deskew.jpg",rotated_deskew)
#cv2.imwrite("rotated median complex.jpg",rotated_median_complex)
#cv2.imwrite("rotated median simple.jpg",rotated_median_simple)
#cv2.imwrite("max contour simple.jpg",max_contour_rotated_simple)
#cv2.imwrite("max contour complex.jpg",max_contour_rotated_complex)
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
#cv2.imshow("sharp_copy",sharp_copy)
#cv2.imshow("negated_erode",negated_erode)
#cv2.imshow("closing",closing)



cv2.waitKey(0)
cv2.destroyAllWindows()