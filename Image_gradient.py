import cv2 as s
import numpy as np
from matplotlib import pyplot as plt

img=s.imread('sudoku.png')
img=s.cvtColor(img,s.COLOR_BGR2RGB)

lap=s.Laplacian(img,s.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))

sobelx=s.Sobel(img,s.CV_64F,1,0)
sobely=s.Sobel(img,s.CV_64F,0,1)

sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

sobelcomb=s.bitwise_or(sobelx,sobely)

titles=['img','laplacian','sobelx','sobely','sobelcombined']
images=[img,lap,sobelx,sobely,sobelcomb]
for i in range(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()