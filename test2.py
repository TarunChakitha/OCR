import cv2
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
img = cv2.imread("deskew-9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalised = cv2.equalizeHist(gray)
#res = np.hstack((gray,equalised))
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(gray)

data = pytesseract.image_to_string(cl1)
print(data)
d = pytesseract.image_to_data(cl1, output_type=Output.DICT)
print(d.keys())

cv2.imshow("gray",gray)
cv2.imshow("equalised",equalised)
cv2.imshow("cl1",cl1)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
