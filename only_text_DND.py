import cv2
import numpy as np 
from matplotlib import pyplot as plt
img = cv2.imread("text1.png")

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-9.jpg'

img = cv2.imread("text1.png")

#img = cv2.pyrDown(large)
img = cv2.resize(img, None, fx = 0.5, fy = 0.5,interpolation=cv2.INTER_LINEAR)
img1 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
gradient_rect = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,kernel_rect)
#gradient_ellipse = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,kernel_ellipse)
#gradient_cross = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,kernel_cross)
#print(kernel)
ret_rect, thresh_rect = cv2.threshold(gradient_rect, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#ret_ellipse, thresh_ellipse = cv2.threshold(gradient_ellipse, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#ret_cross, thresh_cross = cv2.threshold(gradient_cross, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret_rect)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))
#kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,1))
#kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,1))
#print(kernel1)
closing_rect = cv2.morphologyEx(thresh_rect,cv2.MORPH_CLOSE,kernel_rect)
#closing_ellipse = cv2.morphologyEx(thresh_ellipse,cv2.MORPH_CLOSE,kernel_ellipse)
#closing_cross = cv2.morphologyEx(thresh_cross,cv2.MORPH_CLOSE,kernel_cross)
#closing = cv2.erode(closing,np.ones((3,3),np.uint8),iterations=2)
contours, hierarchy = cv2.findContours(closing_rect, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#mask = np.asarray()
#print(mask)
#cv2.imshow("mask",mask)
#print(thresh.shape)
mask = np.zeros(img.shape,np.uint8)

print(len(contours))
for cnt in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[cnt])
    if w>22 and 52>h>13:
        cv2.rectangle(img1,(x,y),(x+y,y+h),(255,0,0),2)
        mask[y:y+h,x:x+w] = 255
        masking = mask & img
#gray2 = cv2.cvtColor(masking, cv2.COLOR_BGR2GRAY)
#masked_thresh_rect = cv2.adaptiveThreshold(gray2,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#print(ret_rect2)
#cv2.imshow("gray",gray)
#cv2.imshow("gradient_cross",gradient_cross)
#cv2.imshow("gradient_ellipse",gradient_ellipse)
cv2.imshow("gradient_rect",gradient_rect)
cv2.imshow("thresh_rect",thresh_rect)
#cv2.imshow("thresh_ellipse",thresh_ellipse)
#cv2.imshow("thresh_cross",thresh_cross)
#cv2.imshow("img1",img1)
#cv2.imshow("closing_rect",closing_rect)
#cv2.imshow("closing_ellipse",closing_ellipse)
#cv2.imshow("closing_cross",closing_cross)
#plt.imshow(closing)
#plt.title("final")
#plt.show()
cv2.imshow("mask",masking)
#cv2.imshow("masking",masking)
#cv2.imshow('masked_thresh_rect',masked_thresh_rect)
cv2.imshow('rects', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

