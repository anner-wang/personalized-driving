from gluoncv import model_zoo,data,utils
from matplotlib import pyplot as plt
import cv2
import time
import numpy as np
focalLength=100
CAR_WIDTH=80
PERSON_WIDTH=20
net=model_zoo.get_model('ssd_512_resnet50_v1_voc',pretrained=True)
im_fname='/home/anner/code/gluoncv/ssd/26.jpg'
x,img=data.transforms.presets.ssd.load_test(im_fname,short=512)
class_IDS,scores,bounding_boxs=net(x)
for i in range(len(class_IDS[0])):
    left_top_x=bounding_boxs[0][i][0].asscalar()
    left_top_y=bounding_boxs[0][i][1].asscalar()
    right_bottom_x=bounding_boxs[0][i][2].asscalar()
    right_bottom_y=bounding_boxs[0][i][3].asscalar()
    height=right_bottom_y-left_top_y
    width=right_bottom_x-left_top_x
    if scores[0][i]>0.5:
        if class_IDS[0][i]==6:
            cv2.rectangle(img   ,(left_top_x,left_top_y),(right_bottom_x,right_bottom_y),(0,0,255),2)
            distance=round(focalLength*CAR_WIDTH/(width),2)
            cv2.putText(img,str(distance)+'cm',(int(left_top_x),int(left_top_y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
	         0.5, (0, 0, 255),1)
        if class_IDS[0][i]==14:
            cv2.rectangle(img,(left_top_x,left_top_y),(right_bottom_x,right_bottom_y),(0,255,0),2)
            distance=round(focalLength*PERSON_WIDTH/(width),2)
            cv2.putText(img,str(distance)+'cm',(int(left_top_x),int(left_top_y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
	         0.4, (0, 255,0),1)
cv2.imshow('image',img)
cv2.waitKey(0)