import cv2
import numpy as np 
from matplotlib import pyplot as plt
img = cv2.imread("text1.png")
img1 = img.copy()
#img = cv2.pyrDown(large)
img = cv2.resize(img, None, fx = 0.5, fy = 0.5,interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,kernel)
print(kernel)
ret, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))
#print(kernel1)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel1)
#closing = cv2.erode(closing,np.ones((3,3),np.uint8),iterations=2)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#mask = np.asarray()
#print(mask)
#cv2.imshow("mask",mask)
#print(thresh.shape)
mask = np.zeros(img.shape,np.uint8)

print(len(contours))
for cnt in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[cnt])
    if w>22 and 52>h>13:
        cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
        mask[y:y+h,x:x+w] = 255
        masking = mask & img

#cv2.imshow("gray",gray)
#cv2.imshow("thresh",thresh)
#cv2.imshow("otsu",otsu)
#cv2.imshow("closing",closing)
#plt.imshow(closing)
#plt.title("final")
#plt.show()
#cv2.imshow("mask",mask)
cv2.imshow("masking",masking)
cv2.imshow('rects', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

