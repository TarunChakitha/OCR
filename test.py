import cv2 
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

image = cv2.imread(path_skew)
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image)
deskewed,angle = IMAGE_TOOLS_LIB.rotate(image,(255,255,255))
image = IMAGE_TOOLS_LIB.image_resize(image,1200,675)
image1 = image.copy()
image2 = image.copy()
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image)
deskewed,angle = IMAGE_TOOLS_LIB.rotate(image,(255,255,255))
#x = cv2.cvtColor(deskewed,cv2.COLOR_BGR2GRAY)
#kernel = cv2.getGaussianKernel(5,0)
#print(kernel)
#gaussian = cv2.filter2D(deskewed,-1,kernel)
gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#gaussian = cv2.GaussianBlur(deskewed,(3,3),0)
#gaussian = cv2.GaussianBlur(deskewed,(7,7),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#print(kernel)
sharp = cv2.filter2D(gaussian,-1,kernel)
#sharp2 = cv2.filter2D(gaussian2,-1,kernel)

#gaussian = cv2.GaussianBlur(sharp,(5,5),0)
#sharp = cv2.filter2D(gaussian,-1,kernel)

gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
#equalised = clahe.apply(gray)
deskewed1 = deskewed.copy()
deskewed2 = deskewed.copy()
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(deskewed)
#print(type(hist_eq))
print(angle)
#no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
#gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#smooth = cv2.addWeighted(gaussian,1.5,deskewed,-0.5,0)
#bilateral = cv2.bilateralFilter(deskewed,9,75,75)
binary_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
binary_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)


cv2.imshow("gaussia",gaussian)
#cv2.imshow("binary",binary)
cv2.imshow("otsu",otsu)
#cv2.imshow("binary_mean",binary_mean)
cv2.imshow("contrast",contrast)
cv2.imshow("result",result)
cv2.imshow("binary_gaussian",binary_gaussian)
cv2.imshow("gray",gray)
cv2.imshow("dilation",dilation)
cv2.imshow("erode",erosion)
cv2.imshow("opening",opening)
cv2.imshow("edges",edges)
cv2.imshow("lap",lap)
cv2.imshow("sharp",sharp)
#cv2.imshow("sharp2",sharp2)
#cv2.imshow("gaussian",gaussian)
#cv2.imshow("smooth",smooth)
#cv2.imshow("deskewd2",deskewed2)
#cv2.imshow("deskewed1",deskewed1)
cv2.imshow("hist_eq",equalised)
#cv2.imshow("image1",image1)
#cv2.imshow("image2",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

