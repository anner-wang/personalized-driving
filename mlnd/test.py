from moviepy.editor import VideoFileClip
from scipy.misc import imresize
import numpy as np
def resize(img):
    print(img.shape)
    return img
clip=VideoFileClip('detection.mp4')
vid_clip=clip.fl_image(resize)
vid_clip.write_videofile('temp.mp4', audio=False)