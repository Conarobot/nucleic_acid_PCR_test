import cv2
import os
import random


def img_to_video(img_dir, fps=8):
    files = os.listdir(img_dir)
    img_path0 = os.path.join(img_dir, files[0])
    img = cv2.imread(img_path0)  # 读取第一张图片
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video_name = img_dir.split('/')[-1] + '.avi'
    videoWrite = cv2.VideoWriter(video_name, fourcc, fps, size)  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    # videoWrite = cv2.VideoWriter('0.mp4',fourcc,fps,(1920,1080))

    out_num = len(files)
    for i in range(0, out_num):
        file_path = os.path.join(img_dir, files[i])
        img = cv2.imread(file_path)
        print(img.shape)
        videoWrite.write(img)


if __name__ == '__main__':
    # data_dir = './difference_position'
    data_dir = './draw_out'
    # data_dir = './difference_direction'
    img_to_video(data_dir, 5)