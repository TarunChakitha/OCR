import cv2
import numpy as np 

img = cv2.imread("captcha1.png")
#img = cv2.resize(img,None,fx = 3,fy = 3,interpolation = cv2.INTER_LINEAR)

#copy of original image
img1 = img.copy()
img2 = img.copy()

# original image
#cv2.imshow("original",img)

# median blur
median_blur = cv2.medianBlur(img,27)
#cv2.imshow("median_blur",median_blur)

# gray scale image
gray= cv2.cvtColor(median_blur,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",gray)

# otsu
ret,otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
#cv2.imshow("otsu",otsu)
#otsu1 = otsu.copy()

# applying bitwise not to invert color
inverted = cv2.bitwise_not(otsu)
#cv2.imshow("inverted",inverted)

# dilation
dilate = cv2.dilate(inverted,np.ones((5,5),np.uint8),iterations = 3)
#cv2.namedWindow("dilate",cv2.WINDOW_NORMAL)
#cv2.imshow("dilate",dilate)

img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
mask = img_gray & dilate
#cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
#cv2.imshow("mask",mask)

# find contours
contours, hierarcky = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# min area rect and boxpoints 
rect = cv2.minAreaRect(max(contours,key = cv2.contourArea))
print(rect)
box = cv2.boxPoints(rect)
print(box)
box = np.int0(box)
rectangle = cv2.drawContours(img,[box],-1,(0,255,0),2)
cv2.imshow("rectangle",rectangle)

# draw points on corners
#r = 5
#for b in box:
#    cv2.circle(img,tuple(b),r,(0,255,0),-1)

#cv2.drawContours(img,[box],0,(0,255,0),2)
#cv2.imshow("detect",img)

def crop(image,center,angle,width,height):
    shape = (image.shape[1],image.shape[0]) # cv2.warpAffine expects shape in (length,height) but image we take has shape in (height , length)
    # Third argument of the cv2.warpAffine() function is the size of the output image, which should be in the form of (width, height).
    #-Remember width = number of columns, and height = number of rows.
    rotation_matrix = cv2.getRotationMatrix2D(center , angle, scale = 1)
    image = cv2.warpAffine(image, rotation_matrix, shape)

    x = int(center[0]-width/2)
    y = int(center[1]-height/2)

    image = image[ y:y+height, x:x+width ]
    return image

cropped = crop(img1, center = rect[0], angle = int(rect[2]), width = int(rect[1][0]), height = int(rect[1][1]))
#cv2.namedWindow("cropped",cv2.WINDOW_NORMAL)
cv2.imshow("cropped",cropped)

# bgr to gray
gray1 = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray1",gray1)

# median blur
median_blur1 = cv2.medianBlur(gray1,23)
cv2.imshow("median_blur1",median_blur1)

# otsu
ret1, otsu1 = cv2.threshold(median_blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret1)
cv2.imshow("otsu1",otsu1)



# applying bitwise not to invert color
inverted1 = cv2.bitwise_not(otsu1)
cv2.imshow("inverted1",inverted1)

# dilation
dilation1 = cv2.dilate(inverted1,np.ones((5,5),np.uint8),iterations=3)

#dilation1 = cv2.morphologyEx(inverted1,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=1)
cv2.imshow("dilation1",dilation1)

# 

mask1 = dilation1 & gray1
cv2.imshow("mask1",mask1)










cv2.waitKey(0)
cv2.destroyAllWindows()
