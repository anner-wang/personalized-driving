import cv2
import os
images_path='data/video_images/'
result_path='data/predictions/'
pre_path='/home/anner/code/segmentation/image-segmentation-keras/'
image_names=[]
result_name=[]
for name in os.listdir(pre_path+images_path):
    image_names.append(pre_path+images_path+name)
for name  in os.listdir(pre_path+result_path):
    result_name.append(pre_path+result_path+name)
image_names.sort()
result_name.sort()
for i in range(len(image_names)):
    cv2.imshow('images',cv2.imread(image_names[i]))
    cv2.imshow('result',cv2.imread(result_name[i]))
    cv2.waitKey(100)
cv2.destroyAllWindows()