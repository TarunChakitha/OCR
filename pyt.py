import numpy as np 
import cv2 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
img = cv2.imread("deskew-9.jpg")
deskewed = IMAGE_TOOLS_LIB.rotate(img,(0,0,0))
data = pytesseract.image_to_string(img)
print(data)
d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())
'''
print(d['text'][12])
print(d['conf'][12])
print(d['left'][12])
print(d['top'][12])
print(d['width'][12])
print(d['height'][12])
'''
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i])>40:
        (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()