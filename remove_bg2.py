import cv2
import numpy as np 

# loading, convert to grayscale, otsu
img = cv2.imread("stop.jpg")
#img = cv2.resize(img,None,fx = 2, fy = 2, interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#print(ret)
cv2.imshow("thresh otsu",thresh)

# Opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
cv2.imshow("opening",opening)

# Finding conturs and removing further noice
cnts = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts)==2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        cv2.drawContours(opening,[c],-1,(0,0,0),-1)
result = 255 - opening
result = cv2.GaussianBlur(result,(3,3),0)
cv2.imshow("result",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
