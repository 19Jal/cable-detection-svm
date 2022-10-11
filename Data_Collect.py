import os
import cv2
import numpy as np
import math
from imutils import paths
from skimage import feature
from sklearn.svm import LinearSVC

capture = cv2.VideoCapture('Video_Kabel.mp4')
success,img = capture.read()
d1 = 'Train/1/'     #Kabel
d2 = 'Train/2/'     #Non-Kabel
n1 = 0
n2 = 0
success, frm = capture.read()
count = 0

for l in range(0,28):
    while success:
        # -- Ambil Sampel untuk Training setiap 30 frame --
        success,frm = capture.read()
        count+=1
        if count%30 == 0 or not success:
            break
    if success:
        # -- Canny Edge Detection --
        gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Gray Level',gray)
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)
        #cv2.imshow('Gaussian Blur',gblur)
        dst = cv2.Canny(gblur, 50, 150)
        #cv2.imshow('Canny Edge Detection',dst,)
        
        # -- Morphological Filter agar Kabel lebih tampak --
        dst = cv2.dilate(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
        dst = cv2.erode(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
        dst = cv2.dilate(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
        dst = cv2.erode(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
        #cv2.imshow('Morphed',dst)

        # -- Probability Hough Line Transform dan
        # Pengambilan Data untuk Training --
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, 20, 160, 8)          
        if linesP is not None:
            for i in range(0, len(linesP)):
                x1,y1,x2,y2 = linesP[i][0]
                p1 = (x1,y1)
                p2 = (x2,y2)
                tmp = frm.copy()
                
                if (y1<y2):
                    if (x1<x2):
                        img = frm[y1:y2,x1:x2]
                    if (x2<x1):
                        img = frm[y1:y2,x2:x1]
                if (y1>y2):
                    if (x1<x2):
                        img = frm[y2:y1,x1:x2]
                    if (x2<x1):
                        img = frm[y2:y1,x2:x1]
                        
                cv2.line(tmp, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.rectangle(tmp,p1,p2,(0,255,0),1)
                cv2.imshow('ROI',img)
                cv2.imshow('FRAME',tmp)
                k = cv2.waitKey(0)
                # ESC -> Exit
                # Space -> Skip frame
                # 0 -> Skip sampel
                # 1 -> Kabel
                # 2 -> Non-Kabel
                if k == 27 : break;
                elif k == 32 : break;
                elif k == 48 : continue;
                elif k == 49 : n1 += 1 ; cv2.imwrite(d1+str(n1)+'.jpg',img)
                elif k == 50 : n2 += 1 ; cv2.imwrite(d2+str(n2)+'.jpg',img)
                cv2.destroyWindow('ROI')
            if k == 27 : break;
capture.release()
cv2.destroyAllWindows()
