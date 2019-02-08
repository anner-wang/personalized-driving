#由于服务器性能限制，需要删去已经预测的照片
import os
pre_path='/home/anner/code/gluoncv/ssd/'
images_path=pre_path+'images/'
pro_images_path=pre_path+'pro_images/'
for name in os.listdir(pro_images_path):
    if os.path.exists(images_path+name):
        os.remove(images_path+name)
        print(images_path+name+' deleted')
    else:
        print('file does not exist')
