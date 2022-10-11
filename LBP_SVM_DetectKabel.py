# Import Modules
import os
import cv2
import numpy as np
from imutils import paths
from skimage import feature
from sklearn.svm import LinearSVC

# Define LBP Class
class LocalBinaryPattern :
    # Variable Initialization
    def __init__(self,points,radius) :
        self.points = points
        self.radius = radius
    # Describe LBP Feature
    def describe(self,image,ep=1e-7) :
        lbp = feature.local_binary_pattern(image,self.points,self.radius,method='uniform')
        htg = np.histogram(lbp.ravel(),bins=np.arange(0,self.points+3),range=(0,self.points+2))[0]
        htg = htg.astype('float') ; htg /= (htg.sum()+ep) ; return htg

# Learning Process
LBP = LocalBinaryPattern(10,1)
data = [] ; lab = []
for p in paths.list_images('Train_3') :
    img = cv2.imread(p)
    gry = cv2.cvtColor(img,6)
    htg = LBP.describe(gry)
    lab.append(p.split(os.path.sep)[-2])
    data.append(htg)
model = LinearSVC(C=100.0,random_state=50,max_iter=50000)
model.fit(data,lab)

# Detection Process
vid = cv2.VideoCapture('Video_Kabel.mp4')
success,frm = vid.read()
while success :
    frm = vid.read()[1]
    # -- Hough Line Transform --
    gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    dst = cv2.Canny(gblur, 50, 150)
    dst = cv2.dilate(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
    dst = cv2.erode(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
    dst = cv2.dilate(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
    dst = cv2.erode(dst,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, 20, 160, 8)
    img = frm.copy()
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            x1,y1,x2,y2 = linesP[i][0]
            p1 = (x1,y1)
            p2 = (x2,y2)
            tmp = frm.copy()
            if (y1<y2):
                if (x1<x2):
                    imr = img[y1:y2,x1:x2]
                if (x2<x1):
                    imr = img[y1:y2,x2:x1]
            elif (y2<y1):
                if (x1<x2):
                    imr = img[y2:y1,x1:x2]
                if (x2<x1):
                    imr = img[y2:y1,x2:x1]     
            #imr = cv2.GaussianBlur(imr,(3,3),0)
            gry = cv2.cvtColor(imr,6)
            htg = LBP.describe(gry)
            out = model.predict(htg.reshape(1,-1))[0]           
            if out == '1' :     #Jika Kabel,
                cv2.line(frm, (x1, y1), (x2, y2), (255,0,0), 2)
                #cv.rectangle(frm,p1,p2,(0,255,0),1)
                    
    cv2.imshow('FRAME',frm)
    if cv2.waitKey(1) == 27 : break
cv2.destroyAllWindows()
