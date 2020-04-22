import cv2
import numpy as np 
from deskew import determine_skew
import DESKEW_LIB 
image = cv2.imread("deskew-12.jpg")
#image = cv2.resize(image,None, fx = 0.25, fy = 0.1, interpolation = cv2.INTER_LINEAR)
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)
angle = determine_skew(gray)
'''
# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
print(coords)
angle = cv2.minAreaRect(coords)[-1]
#rect = cv2.minAreaRect(coords)
#print(rect)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#dst = cv2.drawContours(image,[box],-1,(0,255,0),2)
#cv2.imshow("dst",dst)
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle'''
if angle < -45:
	angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle
# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# draw the correction angle on the image so we can validate it
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
gray1 = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
angle1 = determine_skew(gray1)
print(angle1)
new = DESKEW_LIB.rotate(rotated,angle1,255)

# show the output image
print("angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.imshow("new", new)
cv2.waitKey(0)