import cv2 as cv
import numpy as np
img=cv.imread("life.jpeg")
gry_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
temp=cv.imread("lf.jpeg",0)
w,h=temp.shape[::-1]
res=cv.matchTemplate(gry_img,temp,cv.TM_CCORR_NORMED)
thresold=0.99
loc=np.where(res>=thresold)
for pt in zip (*loc[::-1]):
    cv.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,0),2)
cv.imshow("img",img)
cv.waitKey(0)
cv.destroyAllWindows()