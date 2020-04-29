import cv2 
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output
#from PIL import Image
#from ocr_tesseract_wrapper import OCR
#x = OCR()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'
image = cv2.imread(path_skew)
image_resized = IMAGE_TOOLS_LIB.image_resize(image,1200,675)
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image_resized)
deskewed, angle = IMAGE_TOOLS_LIB.rotate(no_shadows,(255,255,255))

contrast = cv2.convertScaleAbs(deskewed,alpha=1.3,beta=40)
gamma = 1
lookUpTable = np.empty((1,256),np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i/255,gamma)*255,0,255)
result = cv2.LUT(deskewed,lookUpTable)


gaussian = cv2.GaussianBlur(result,(5,5),0)
#gaussian = cv2.GaussianBlur(deskewed,(3,3),0)
#gaussian = cv2.GaussianBlur(deskewed,(7,7),0)
#edges = cv2.Canny(deskewed,100,200)
kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#print(kernel)
sharp = cv2.filter2D(gaussian,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
equalised = clahe.apply(gray)

binary_gaussian = cv2.adaptiveThreshold(equalised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(equalised,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
edges = cv2.Canny(result,100,200)
lap = cv2.Laplacian(result,cv2.CV_16S,3)
lap = cv2.convertScaleAbs(lap)
dilation = cv2.dilate(binary_gaussian,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
erosion = cv2.erode(dilation,np.ones((3,3),np.uint8))
opening = cv2.morphologyEx(binary_gaussian,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))




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

