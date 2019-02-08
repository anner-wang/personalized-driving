from moviepy.editor import VideoFileClip
from scipy.misc import imresize
import numpy as np
import cv2
import time
count=0
def video2image(img):
    global count
    img=np.array(img)
    cv2.imwrite('images/'+str(count)+'.jpg',img)
    count+=1
    print(time.time())
    return img
clip=VideoFileClip('test.mp4')
vid_clip=clip.fl_image(video2image)
vid_clip.write_videofile('temp.mp4', audio=False)