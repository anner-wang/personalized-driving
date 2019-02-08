#删除已经处理的文件
import os
import time
import cv2
origin_path='/home/anner/code/lanenet-lane-detection/data/tusimple_test_image/origin_images'
process_path='/home/anner/code/lanenet-lane-detection/data/tusimple_test_image/process_images'
images=os.listdir(process_path)
for image_name in images:
    path=origin_path+'/'+image_name
    if os.path.exists(path):
        os.remove(path)
        print(path)
    else:
        print('wrong')
