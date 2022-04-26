import numpy as np
import cv2
import os
import csv
if not os.path.exists("Test"):
    os.makedirs("Test")
if not os.path.exists("Test/train"):
    os.makedirs("Test/train")
if not os.path.exists("Test/test"):
    os.makedirs("Test/test")
path="Test"
path1 = "Test"
a=['label']

for i in range(64*64):
    a.append("pixel"+str(i))
    

#outputLine = a.tolist()


label=0
var = 0
c1 = 0
c2 = 0

for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
       	    if not os.path.exists(path1+"/train/"+dirname):
                os.makedirs(path1+"/train/"+dirname)
            if not os.path.exists(path1+"/test/"+dirname):
                os.makedirs(path1+"/test/"+dirname)
            num=0.75*len(files)
            #num = 100
            i=0
            for file in files:
                var+=1
                actual_path=path+"/"+dirname+"/"+file
                actual_path1=path1+"/"+"train/"+dirname+"/"+file
                actual_path2=path1+"/"+"test/"+dirname+"/"+file
                img = cv2.imread(actual_path, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(5,5),2)

                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                if i<num:
                    c1 += 1
                    cv2.imwrite(actual_path1 , res)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2 , res)
                    
                i=i+1
                
        label=label+1
print(var)
print(c1)
print(c2)





