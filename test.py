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
#contours_otsu,hierarchy = cv2.findContours(double_opening_dilated_7x7,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours , hierarchy = cv2.findContours(double_opening_dilated_7x7_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea)
max_contour = contours[-1]
rect = cv2.minAreaRect(max_contour)
(y1,x1),(y2,x2) = rect[:2]
max_contour_height = abs(y2-y1)
max_contour_width = abs(x2-x1)
max_contour_angle = rect[-1]
box = cv2.boxPoints(rect)
box = np.int0(box)
print("box")
print("\n")
print(box)
cv2.drawContours(image_resized1,[box],0,(0,255,0),2)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
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

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

pts = np.array((box), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)


cv2.imwrite("max area contour.jpg",image_resized1)
cv2.imwrite("warped.jpg",warped)
