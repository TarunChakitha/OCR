import numpy as np 
import cv2 
from deskew import determine_skew, determine_skew_dev
from typing import Tuple, Union
import math

image = cv2.imread("deskew-22.jpg")
image = cv2.resize(image,None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_LINEAR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#ret,thresh = cv2.threshold(gray, 50, 255,cv2.THRESH_BINARY )
cv2.imshow("thresh",thresh)
print(ret)

angle = determine_skew(thresh)
print(angle)

def rotate(image: np.ndarray, ang8e: float, background_color):
    old_width, old_height = image.shape[:2]
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
rotated = rotate(image,angle,(0,0,0))

#gray2 = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
#angle1 = determine_skew(gray2)
#rotated1 = rotate(rotated,angle1,(0,0,0))

cv2.imshow("image",image)
cv2.imshow("rotated",rotated)
#cv2.imshow("rotated1",rotated1)
cv2.waitKey(0)
cv2.destroyAllWindows()
