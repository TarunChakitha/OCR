import cv2 
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output
from PIL import Image
from ocr_tesseract_wrapper import OCR
x = OCR()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

image = cv2.imread(path_skew)
image = IMAGE_TOOLS_LIB.image_resize(image,1200,675)
image1 = image.copy()
image2 = image.copy()
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image)
gaussian = cv2.GaussianBlur(no_shadows,(5,5),0)
#gaussian = cv2.GaussianBlur(gaussian,(3,3),0)
#gaussian2 = cv2.GaussianBlur(no_shadows,(7,7),0)



kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#print(kernel)
sharp = cv2.filter2D(gaussian,-1,kernel)
#sharp2 = cv2.filter2D(gaussian2,-1,kernel)

deskewed,angle = IMAGE_TOOLS_LIB.rotate(sharp,(255,255,255))
deskewed = cv2.cvtColor(deskewed,cv2.COLOR_BGR2GRAY)
deskewed1 = deskewed.copy()
deskewed2 = deskewed.copy()
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(deskewed)
#print(type(hist_eq))

#no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
#gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#smooth = cv2.addWeighted(gaussian,1.5,deskewed,-0.5,0)
#bilateral = cv2.bilateralFilter(deskewed,9,75,75)
binary = cv2.adaptiveThreshold(deskewed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret, otsu = cv2.threshold(deskewed,30,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)
#print(ret)
#bilateral = cv2.bilateralFilter(binary,9,75,75)
custom_oem_psm_config1 = r'--oem 3 --psm 11'
custom_oem_psm_config2 = r'--oem 3 --psm 12'
d1 = pytesseract.image_to_data(deskewed, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
d2 = pytesseract.image_to_data(deskewed, output_type=Output.DICT,config=custom_oem_psm_config2,lang='eng')
#print(d.keys())
print(len(d1['text']))
print(len(d2['text']))
n_boxes1 = len(d1['text'])
n_boxes2 = len(d2['text'])

for i in range(n_boxes1):
    if int(d1['conf'][i])>60:
        print(d1['text'][i])
        (x,y,w,h) = (d1['left'][i],d1['top'][i],d1['width'][i],d1['height'][i])
        cv2.rectangle(deskewed1,(x,y),(x+w,y+h),(0,255,0),1)


for i in range(n_boxes2):
    if int(d2['conf'][i])>60:
        print(d2['text'][i])
        (x,y,w,h) = (d2['left'][i],d2['top'][i],d2['width'][i],d2['height'][i])
        cv2.rectangle(deskewed2,(x,y),(x+w,y+h),(0,255,0),1)

cv2.imshow("gaussia",gaussian)
cv2.imshow("binary",binary)
cv2.imshow("otsu",otsu)
#cv2.imshow("bilateral",bilateral)
cv2.imshow("sharp",sharp)
#cv2.imshow("sharp2",sharp2)
#cv2.imshow("gaussian",gaussian)
#cv2.imshow("smooth",smooth)
cv2.imshow("deskewd2",deskewed2)
cv2.imshow("deskewed1",deskewed1)
#cv2.imshow("hist_eq",cl1)
#cv2.imshow("image1",image1)
#cv2.imshow("image2",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



