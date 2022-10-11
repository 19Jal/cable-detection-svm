# Import Modules
import os
import cv2 as cv
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
for p in paths.list_images('Train') :
    img = cv.imread(p)
    gry = cv.cvtColor(img,6)
    htg = LBP.describe(gry)
    lab.append(p.split(os.path.sep)[-2])
    data.append(htg)
model = LinearSVC(C=100.0,random_state=50,max_iter=50000)
model.fit(data,lab)

# Detection Process
vid = cv.VideoCapture('Video_Kabel.mp4')
vid.set(1,0) ; size = (480,850)
roix,roiy = 100,100
while True :
    frm = vid.read()[1]
    #frm = cv.cvtColor(frm,cv.COLOR_BGR2GRAY)
    #frm = cv.GaussianBlur(frm, (5, 5), 0)
    #frm = cv.Canny(frm, 50, 150)
    img = frm.copy()
    
    for i in range(int(size[0]/roiy)) :
        for j in range(int(size[1]/roix)) :
            p1 = (j*roix,i*roiy)
            p2 = (j*roix+roix,i*roiy+roiy)
            imr = img[p1[1]:p2[1],p1[0]:p2[0]]
            imr = cv.GaussianBlur(imr,(3,3),0)
            gry = cv.cvtColor(imr,6)
            htg = LBP.describe(gry)
            out = model.predict(htg.reshape(1,-1))[0]
            if out == '1' : cv.rectangle(frm,p1,p2,(0,255,0),1)
            # 1 = Kabel
            # 2 = Tiang
            # 3 = Jalan
            # 4 = Pohon
            # 5 = Danau
    cv.imshow('FRAME',frm)
    if cv.waitKey(1) == 27 : break
cv.destroyAllWindows()
