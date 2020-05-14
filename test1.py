import numpy as np 
import cv2 
import math
import pyttsx3
import pytesseract
import IMAGE_TOOLS_LIB
from deskew import determine_skew
from typing import Tuple, Union
from pytesseract import Output
from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-19.jpg'

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

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = np.uint8)
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def get_width_height(box):
    contour_height = math.sqrt((pow((box[0][0] - box[1][0]),2))+(pow((box[0][1] - box[1][1]),2)))
    contour_width = math.sqrt((pow((box[0][0] - box[3][0]),2))+(pow((box[0][1] - box[3][1]),2)))
    return contour_width,contour_height


'''
    for median angles
                        '''

heights = []
widths = []
angles = []
contours_otsu,hierarchy = cv2.findContours(double_opening_dilated_7x7,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in range(len(contours_otsu)):
    rect = cv2.minAreaRect(contours_otsu[cnt])    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = order_points(box)
    width, height = get_width_height(box)
    widths.append(width)
    heights.append(height)
    angles.append(rect[-1])

median_angles = np.median(angles)
mean_heights = np.mean(heights)
mean_widths = np.mean(widths)
median_heights = np.median(heights)
median_widths = np.median(widths)
print(median_angles)
print(mean_heights,median_heights)
print(mean_widths,median_widths)
'''
    for max contour
                    '''

contours , hierarchy = cv2.findContours(double_opening_dilated_7x7_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea)
max_contour = contours[-1]
rect = cv2.minAreaRect(max_contour)
max_contour_angle = rect[-1]
box = cv2.boxPoints(rect)
box = np.int0(box)
print("box")
print("\n")
#print(rect[:2])
print(box)
ordered = order_points(box)
print("orederd")
print(ordered)
max_contour_height = math.sqrt((pow((box[0][0] - box[1][0]),2))+(pow((box[0][1] - box[1][1]),2)))
max_contour_width = math.sqrt((pow((box[0][0] - box[3][0]),2))+(pow((box[0][1] - box[3][1]),2)))
print(max_contour_width,max_contour_height)
cv2.drawContours(image_resized1,[box],0,(0,255,0),2)

def corrected_angle(angle,contour_width,contour_height):
    if (contour_width < contour_height):
        print("width less than height")
        corrected_angle = angle - 90
    else:
        print("width greater than height")
        corrected_angle = angle
    if ( 0 <= angle <= 90):
        print("angle in 4th quadrant")
        corrected_angle = -angle
    return corrected_angle

print("deskew corrected angle",determine_skew(gray))
print("angle deteccted",rect[-1])
print("corrected angle",corrected_angle(max_contour_angle,max_contour_width,max_contour_height))
print("corrected median angle",corrected_angle(median_angles,mean_widths,mean_heights))

def rotate(image: np.ndarray, angle, background_color): # OFFIAL DOCUMENTATION
    old_width, old_height = image.shape[:2]
    gray_scale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    #print(old_width,old_height)
    #print(width,height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  
    #print(rot_mat[1, 2],rot_mat[0, 2])
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    #print(rot_mat[1, 2],rot_mat[0, 2])
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background_color)

rotated_max = rotate(image,corrected_angle(max_contour_angle,max_contour_width,max_contour_height),(0,0,0))
rotated_median = rotate(image,corrected_angle(max_contour_angle,mean_widths,mean_heights),(0,0,0))
cv2.imwrite("rotated median.jpg",rotated_median)
cv2.imwrite("rotated max contour.jpg",rotated_max)