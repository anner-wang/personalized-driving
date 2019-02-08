from moviepy.editor import VideoFileClip
from scipy.misc import imresize
import numpy as np
import cv2
import time
count=0
images_path='/home/anner/code/gluoncv/ssd/pro_images/'
def video2image(img):
    global count
    img=cv2.imread(images_path+str(count)+'.jpg')
    #img=cv2.resize(img,(1800,800))
    count+=1
    return img
clip=VideoFileClip('test.mp4')
vid_clip=clip.fl_image(video2image)
vid_clip.write_videofile('detection.mp4', audio=False)
