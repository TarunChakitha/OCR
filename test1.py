import numpy as np 
import cv2 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-9.jpg'
img = cv2.imread(path_skew)

resized = IMAGE_TOOLS_LIB.image_resize(img, width = 480, height= 640)
no_shadows = IMAGE_TOOLS_LIB.remove_shadows(resized)
deskewed = IMAGE_TOOLS_LIB.rotate(no_shadows,(0,0,0))
print(deskewed[1])

gray_scale = cv2.cvtColor(no_shadows,cv2.COLOR_BGR2GRAY)
#contours, _ = cv2.findContours(gray_scale,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours))
#rects = cv2.minAreaRect(contours[10])
#print(type(rects))
#print(rects)
#data = pytesseract.image_to_data(deskewed)
#print(type(data))
#print(data)

#no_shadows = IMAGE_TOOLS_LIB.remove_shadows(cl1)
#ret, binary = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#binary = cv2.adaptiveThreshold(gray_scale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2 )
#median_blur = cv2.medianBlur(binary,3)
#binary = cv2.adaptiveThreshold(no_shadows,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,2 )
#print(ret)
#closing = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, np.ones(1,1),np.uint8))
#opening = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

#data = pytesseract.image_to_string(cl1)
#print(data)
custom_oem_psm_config = r'--oem 3 --psm 3'
d = pytesseract.image_to_data(img, output_type=Output.DICT,config=custom_oem_psm_config,lang='eng')
print(d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i])>60:
        (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
        rect = cv2.rectangle(deskewed[0],(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("resized",resized)
cv2.imshow("no_shadows",no_shadows)
cv2.imshow("deskewed",deskewed[0])
cv2.imshow("gray_scale",gray_scale)
#cv2.imshow("lab",lab)
#cv2.imshow("binary",binary)
#cv2.imshow("opening",opening)
#cv2.imshow("closing",closing)
#cv2.imshow("cl1",cl1)
#cv2.imshow("median_blur",median_blur)
cv2.imshow("rect",rect)
cv2.waitKey(0)
cv2.destroyAllWindows()
