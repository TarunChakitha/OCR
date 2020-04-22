import numpy as np 
import cv2

img = cv2.imread("stop.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

bilateral = cv2.bilateralFilter(binary,9,75,75)
#median_blur = cv2.medianBlur(bilateral,11)
invert = cv2.bitwise_not(bilateral)

opening = cv2.morphologyEx(invert,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
dilate_opening = cv2.erode(opening, np.ones((13,13),np.uint8),iterations=2)
#dilate_bilateral = cv2.dilate(bilateral, np.ones((5,5),np.uint8),iterations=3)
contours, heirarchy = cv2.findContours(dilate_opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rect = cv2.minAreaRect(max(contours,key = cv2.contourArea))
print(rect)
box = cv2.boxPoints(rect)
print(box)
box = np.int0(box)
rectangle = cv2.drawContours(img,[box],-1,(0,255,0),2)
cv2.imshow("rectangle",rectangle)


cv2.imshow("bilateral",bilateral)
#cv2.imshow("invert",invert)
cv2.imshow("dilate_opening",dilate_opening)
#cv2.imshow("dilatedilate_bilateral",dilate_bilateral)
#cv2.imshow("median_blur",median_blur)
#cv2.imshow("opening",opening)
#cv2.imshow("binary",binary)
#cv2.imshow("img",img)
#cv2.imshow("gray",gray)

cv2.waitKey(0)
cv2.destroyAllWindows()