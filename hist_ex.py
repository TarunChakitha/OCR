import cv2
import numpy as np
from matplotlib import pyplot as plt 
img = cv2.imread("stop.jpg")
cv2.imshow("img",img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv",hsv)
gray = cv2.cvtColor(hsv,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY )
print(ret)
cv2.imshow("binary",binary)
#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#plt.hist(img.ravel(),256,[0,256])
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()