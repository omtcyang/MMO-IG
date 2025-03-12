import cv2
import os.path
import os
import numpy as np
 
 
def img_resize(img):
    height, width = img.shape[0], img.shape[1]
    # 设置新的图片分辨率框架,这里设置为长边像素大小为512
    width_new = 512
    height_new = 512
    # 判断图片的长宽比率
    #if width / height >= width_new / height_new:
    #    img_new = cv2.resize(img, (width_new, int(height * width_new / width)))
    #else:
    #    img_new = cv2.resize(img, (int(width * height_new / height), height_new))
    img_new = cv2.resize(img, (width_new, height_new))
    return img_new
 
 
def read_path(file_path,save_path):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_path):
        # print(filename)
        img = cv2.imread(file_path+'/'+ filename)
        if img is None :
            print("图片更改完毕")
            break
        ####change to size
        image = img_resize(img)
        cv2.imwrite(save_path + filename, image)
 
 
#读取的目录
if __name__ == '__main__':
    file_path = "images800"
    save_path = "images/"
    read_path(file_path,save_path)