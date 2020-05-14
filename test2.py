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
image = cv2.imread("bw2.png")
image1 = image.copy()
#no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image)
#deskewed, angle = IMAGE_TOOLS_LIB.rotate(no_shadows,(255,255,255))
#image_resized = IMAGE_TOOLS_LIB.image_resize(no_shadows,1200,675)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
adaptive_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
custom_oem_psm_config1 = r'--oem 3 --psm 12'
#custom_oem_psm_config2 = r'--oem 1 --psm 12'
#custom_oem_psm_config2 = r'--oem 3 --psm 11'
negated_gray = ~gray
#ocr1 = pytesseract.image_to_data(gray1, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
ocr = pytesseract.image_to_data(gray, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#ocr3= pytesseract.image_to_data(gray3, output_type=Output.DICT,config=custom_oem_psm_config2,lang='eng')
#print(d.keys())
print(len(ocr['text']))
print(ocr['text'])
boxes = len(ocr['text'])
#engine = pyttsx3.init()
for i in range(boxes):
    if int(ocr['conf'][i])>60:
        print(ocr['text'][i])
        (x,y,w,h) = (ocr['left'][i],ocr['top'][i],ocr['width'][i],ocr['height'][i])
        cv2.rectangle(image1,(x,y),(x+w,y+h),(0,0,255),5)
        #cv2.imshow("text",erode)
        #engine.say(d2['text'][i])
        #engine.runAndWait()
        #cv2.waitKey(10)
cv2.imshow("adaptive_gaussian",adaptive_gaussian)
cv2.imshow("~gray",~gray)
cv2.imshow("otsu",otsu)
cv2.imshow("image1",image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

