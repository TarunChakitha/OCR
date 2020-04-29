import cv2 
import numpy as np 
import IMAGE_TOOLS_LIB
import pytesseract
from pytesseract import Output
import pyttsx3

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

image = cv2.imread(path_skew)
#image1 = IMAGE_TOOLS_LIB.image_resize(image,640,480)
image2 = IMAGE_TOOLS_LIB.image_resize(image,1200,675)
#image3 = IMAGE_TOOLS_LIB.image_resize(image,900,500)
#no_shadows1 = IMAGE_TOOLS_LIB.remove_shadows(image1)
no_shadows2 = IMAGE_TOOLS_LIB.remove_shadows(image2)
#no_shadows3 = IMAGE_TOOLS_LIB.remove_shadows(image3)


#deskewed1,angle1 = IMAGE_TOOLS_LIB.rotate(image1,(255,255,255))


deskewed2,angle1 = IMAGE_TOOLS_LIB.rotate(image2,(255,255,255))


#deskewed3,angle3 = IMAGE_TOOLS_LIB.rotate(image3,(255,255,255))

#x = cv2.cvtColor(deskewed,cv2.COLOR_BGR2GRAY)
#kernel = cv2.getGaussianKernel(5,0)
#print(kernel)
#gaussian = cv2.filter2D(deskewed,-1,kernel)
#gaussian1 = cv2.GaussianBlur(deskewed1,(5,5),0)
gaussian2 = cv2.GaussianBlur(deskewed2,(5,5),0)
#gaussian3 = cv2.GaussianBlur(deskewed3,(5,5),0)
#gaussian = cv2.GaussianBlur(deskewed,(3,3),0)
#gaussian = cv2.GaussianBlur(deskewed,(7,7),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#print(kernel)
#sharp1 = cv2.filter2D(gaussian1,-1,kernel)
sharp2 = cv2.filter2D(gaussian2,-1,kernel)
#sharp3 = cv2.filter2D(gaussian3,-1,kernel)
#sharp2 = cv2.filter2D(gaussian2,-1,kernel)

#gaussian = cv2.GaussianBlur(sharp,(5,5),0)
#sharp = cv2.filter2D(gaussian,-1,kernel)

#gray1 = cv2.cvtColor(sharp1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(sharp2,cv2.COLOR_BGR2GRAY)
#gray3 = cv2.cvtColor(sharp3,cv2.COLOR_BGR2GRAY)
#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
#equalised = clahe.apply(gray)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(deskewed)
#print(type(hist_eq))
#print(angle)
#no_shadows = IMAGE_TOOLS_LIB.remove_shadows(deskewed)
#gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#smooth = cv2.addWeighted(gaussian,1.5,deskewed,-0.5,0)
#bilateral = cv2.bilateralFilter(deskewed,9,75,75)
binary_gaussian = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
binary_mean = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray2,180,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#print(ret)
#dilate = cv2.dilate(binary_gaussian,np.ones((3,3),np.uint8))
erode = cv2.erode(binary_gaussian,np.ones((3,3),np.uint8))
#closing = cv2.morphologyEx(binary_gaussian,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
custom_oem_psm_config1 = r'--oem 1 --psm 12'
#custom_oem_psm_config2 = r'--oem 3 --psm 12'
#d1 = pytesseract.image_to_data(gray1, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
d2 = pytesseract.image_to_data(erode, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#d3 = pytesseract.image_to_data(gray3, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#print(d.keys())
#print(len(d1['text']))
print(len(d2['text']))
#print(len(d3['text']))

#n_boxes1 = len(d1['text'])
n_boxes2 = len(d2['text'])
#n_boxes3 = len(d3['text'])
'''
for i in range(n_boxes1):
    if int(d1['conf'][i])>60:
        #print(d1['text'][i])
        (x,y,w,h) = (d1['left'][i],d1['top'][i],d1['width'][i],d1['height'][i])
        cv2.rectangle(deskewed1,(x,y),(x+w,y+h),(0,0,0),2)
'''
engine = pyttsx3.init()
for i in range(n_boxes2):
    if int(d2['conf'][i])>60:
        #print(d2['text'][i])
        (x,y,w,h) = (d2['left'][i],d2['top'][i],d2['width'][i],d2['height'][i])
        cv2.rectangle(erode,(x,y),(x+w,y+h),(0,0,255),1)
        #cv2.imshow("text",erode)
        #engine.say(d2['text'][i])
        #engine.runAndWait()
        #cv2.waitKey(10)
'''
for i in range(n_boxes3):
    if int(d3['conf'][i])>60:
        #print(d2['text'][i])
        (x,y,w,h) = (d3['left'][i],d3['top'][i],d3['width'][i],d3['height'][i])
        cv2.rectangle(deskewed3,(x,y),(x+w,y+h),(0,0,0),2)

cv2.imwrite("binary_mean.jpg",binary_mean)
cv2.imwrite("binary_gaussian.jpg",binary_gaussian)
cv2.imwrite("otsu.jpg",otsu)
cv2.imwrite("gray2.jpg",gray2)
cv2.imwrite("deskewed2.jpg",deskewed2)'''
cv2.imwrite("erode.jpg",erode)
#cv2.imwrite("dilate.jpg",dilate)
#cv2.imwrite("erode.jpg",erode)
#cv2.imwrite("closing.jpg",closing)


#cv2.imshow("gaussia",gaussian)
#cv2.imshow("equalised",equalised)
#cv2.imshow("binary_mean",binary_mean)
#cv2.imshow("binary_gaussian",binary_gaussian)
#cv2.imshow("dilate",dilate)
#cv2.imshow("erode",erode)
#cv2.imshow("otsu",otsu)
#cv2.imshow("gray",gray2)
#cv2.imshow("sharp",sharp)
#cv2.imshow("sharp2",sharp2)
#cv2.imshow("gaussian",gaussian)
#cv2.imshow("smooth",smooth)
#cv2.imshow("900x500",deskewed3)
#cv2.imshow("1200x675",deskewed2)
#cv2.imshow("640X480",deskewed1)
#cv2.imshow("hist_eq",cl1)
#cv2.imshow("image1",image1)
#cv2.imshow("image2",image2)
'''
engine = pyttsx3.init()
space = " "
def getstring(list):
    string = space.join(list)
    return string
voice = getstring(d2['text'])
engine.say(voice)
engine.runAndWait()
'''
cv2.waitKey(0)
cv2.destroyAllWindows()



