import cv2
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-12.jpg'

image = cv2.imread(path_skew)
image1 = image.copy()
resized = IMAGE_TOOLS_LIB.image_resize(image, width = 480, height= 640)
gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
deskewed = IMAGE_TOOLS_LIB.rotate(resized,(0,0,0))
print(deskewed[1])
rect = deskewed[0].copy()

custom_oem_psm_config = r'--oem 3 --psm 11'
d = pytesseract.image_to_data(deskewed[0], output_type=Output.DICT,config=custom_oem_psm_config,lang='eng')
print(d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i])>60:
        print(d['text'][i])
        (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
        rect = cv2.rectangle(rect,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("deskewd",deskewed[0])
cv2.imshow("rect",rect)
cv2.waitKey(0)
cv2.destroyAllWindows()



