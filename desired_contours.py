import cv2
import numpy as np 

image = cv2.imread("skew.jpg")
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
cv2.imshow("thresh",thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,kernel)
print(kernel)
ret, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))
#print(kernel1)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel1)
#closing = cv2.erode(closing,np.ones((3,3),np.uint8),iterations=2)