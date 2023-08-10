import sys, os, time
import math
import numpy as np
import cv2
from scipy import fftpack
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# 返回值:
# static_mm_per_pixel # 每像素表示的毫米长度 
# init_kp_2_position_mm # 关键点的初始位置 n*2 单位：mm  
# time_field_depth_mm # 深度场 w*h
# time_kp_3_displacement_mm # 关键点的位移 n*3 单位：mm 
# time_kp_3_force_N # 关键点的外力 n*3 单位：N

class finger_core_alg:
    def __init__(self, calibration_path, background_path, mask_path, finger_type=1):
        self.finger_type = finger_type
        self.nsz = 40
        # mask图
        mask_read = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_read)
        self.roi = [stats[1, 0], stats[1, 1], stats[1, 2] // 2 * 2, stats[1, 3] // 2 * 2]
        self.mask = mask_read[self.roi[1]:self.roi[1] + self.roi[3],
                    self.roi[0]:self.roi[0] + self.roi[2]]  # get the cropped mask(by JadeCong)
        # 背景图
        bkg_read = cv2.imread(background_path)  # backgroud
        self.bkg = bkg_read[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
        self.bkg_float = self.bkg / 255
        self.bkg_gray = cv2.cvtColor(self.bkg, cv2.COLOR_BGR2GRAY)
        self.bkg = self.bkg.astype(np.int16)
        self.bkg_mask = np.zeros_like(self.bkg_gray)
        if self.finger_type == 1:
            binary = cv2.adaptiveThreshold(self.bkg_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.bkg_mask = cv2.erode(binary, kernel, iterations=1)  # .astype(np.float32)/255
            bkg_mask = self.bkg_mask.copy()  # cv2.erode(binary, kernel, iterations=1)#.astype(np.float32)/255
            bkg_mask[self.mask < 125] = 255
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - bkg_mask)
            self.init_kp_2_position_pixel = np.asarray(
                [[round(cent[0]), round(cent[1])] for stat, cent in zip(stats, centroids) if
                 stat[4] > self.nsz and stat[4] < self.nsz * 8 and cent[0] > 0 and cent[0] < binary.shape[0] and cent[
                     1] > 0 and cent[1] < binary.shape[1]])
            self.tree0 = cKDTree(self.init_kp_2_position_pixel, leafsize=2) if (
                        len(self.init_kp_2_position_pixel) > 10) else []
            self.normdst0 = np.mean([np.mean(self.tree0.query(pt, 9)[0][1:]) for pt in
                                     self.init_kp_2_position_pixel])  # average distance of neigbors
        # 标定数据
        color_set = []  # calibration
        mean_var_count_set = []
        f = open(calibration_path)
        for line in f:
            contents = line.split(":")
            if contents[0] == "pixel_to_mm_scale":
                self.static_mm_per_pixel = float(contents[1])
            elif contents[0] == "image_size":
                image_size = [int(x) for x in contents[1].split(",")]
            elif contents[0] == "n_summary_pt":
                data = [float(x) for x in contents[1].split(",")]
                color_set.append(data[:3])
                mean_var_count_set.append(data[3:6])
        # todo 转为以w,h的中间为中心，这个需要改进为更好的中心
        if self.finger_type == 1:
            self.init_kp_2_position_mm = (self.init_kp_2_position_pixel - [self.roi[3] // 2,
                                                                           self.roi[2] // 2]) * self.static_mm_per_pixel
        binsnum = 50
        mapping = np.zeros((binsnum, binsnum, binsnum, 3), dtype=np.float32)
        mr, mg, mb = np.meshgrid(np.arange(binsnum), np.arange(binsnum), np.arange(binsnum))
        mrf = (mr.astype(np.float32) * 2 / binsnum - 1.0).reshape(binsnum, binsnum, binsnum, 1)
        mgf = (mg.astype(np.float32) * 2 / binsnum - 1.0).reshape(binsnum, binsnum, binsnum, 1)
        mbf = (mb.astype(np.float32) * 2 / binsnum - 1.0).reshape(binsnum, binsnum, binsnum, 1)
        mrgbf = np.concatenate((mgf, mrf, mbf), axis=3)  # the order must be mgf->mrf->mbf
        pts_distances, pts_inds = cKDTree(color_set).query(mrgbf, k=2)
        flat_inds = pts_inds.reshape(-1, pts_inds.shape[-1])
        normals = np.asarray(mean_var_count_set)[flat_inds, :]
        mean_norm = np.mean(normals, axis=1).reshape(binsnum, binsnum, binsnum, 3)
        self.mat_norm = mean_norm  # , image_size, color_set
        # 泊松求解初始值
        (x, y) = np.meshgrid(range(1, self.roi[2] - 2 + 1), range(1, self.roi[3] - 2 + 1), copy=True)
        self.denom = (2 * np.cos(math.pi * x / (self.roi[2] - 2 + 2)) - 2) + (
                    2 * np.cos(math.pi * y / (self.roi[3] - 2 + 2)) - 2)
        (x, y) = np.meshgrid(range(1, (self.roi[2] - 2 + 1) // 2 + 1), range(1, (self.roi[3] - 2 + 1) // 2 + 1),
                             copy=True)
        self.denom2 = (2 * np.cos(math.pi * x / ((self.roi[2] - 2 + 2) // 2)) - 2) + (
                    2 * np.cos(math.pi * y / ((self.roi[3] - 2 + 2) // 2)) - 2)
        # 返回值
        if self.finger_type == 1:
            self.time_kp_3_displacement_mm = np.zeros((len(self.init_kp_2_position_pixel), 3), np.float32)
            self.time_kp_3_force_N = np.zeros_like(self.time_kp_3_displacement_mm)
        self.time_field_depth_mm = np.zeros_like(self.bkg_gray, np.float32)
        return

    def calc(self, img_read, type=0):
        t0 = time.time()
        self.img = img_read[self.roi[1]:self.roi[1] + self.roi[3],
                   self.roi[0]:self.roi[0] + self.roi[2]]  # get region of interest
        t1 = time.time()
        self.time_field_depth_mm.fill(0.0)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 32)
        img_binary = cv2.erode(img_binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        img_binary_mask = img_binary // 255
        if self.finger_type == 1:
            img_binary_mask[self.bkg_mask < 130] = 0  # 除了当前时刻的黑点，也要去掉初始状态的黑点
        binsnum_2 = 25  # 50 #计算区域场的深度值
        obj = (self.img.astype(np.int16) - self.bkg) * binsnum_2 // 255 + binsnum_2
        obj[obj >= 50] = 49
        normals_image = self.mat_norm[obj[..., 0], obj[..., 1], obj[..., 2]]
        # cv2.imshow('normals_image', normals_image)
        normals_image[:, :, 0] *= img_binary_mask
        normals_image[:, :, 1] *= img_binary_mask
        if (type == 1):
            self.time_field_depth_mm = self.poisson_reconstruct(normals_image, self.denom)
        else:
            self.time_field_depth_mm = self.poisson_reconstruct2(normals_image, self.denom2)
        self.time_field_depth_mm *= -self.static_mm_per_pixel
        t2 = time.time()
        # 计算关键点位移
        if self.finger_type == 1:
            self.kp_position1 = self.init_kp_2_position_pixel.copy()
            self.time_kp_3_displacement_mm.fill(0.0)
            img_binary[self.mask < 125] = 255  # mask部分处理
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - img_binary)
            pnts1 = np.asarray([[round(cent[0]), round(cent[1])] for stat, cent in zip(stats, centroids) if
                                stat[4] > self.nsz and stat[4] < self.nsz * 8 and cent[0] > 0 and cent[0] <
                                img_binary.shape[0] and cent[1] > 0 and cent[1] < img_binary.shape[1]])
            pnts1len = len(pnts1)
            if (pnts1len > 10):
                # tree1 = cKDTree(pnts1, leafsize=2)
                costmat = np.zeros((pnts1len, len(self.init_kp_2_position_pixel)), dtype=np.float32)
                for pi in range(pnts1len):
                    dst, idx = self.tree0.query(pnts1[pi], k=8)
                    costmat[pi, idx] = np.exp(-dst / (self.normdst0 * 1))  # costmat[pi, idx] = np.exp(-dst/(normdst*8))
                rowid, colid = linear_sum_assignment(costmat, True)
                pair = np.asarray([[ri, ci] for ri, ci in zip(rowid, colid) if costmat[ri, ci] >= np.exp(-0.5)])
                for p in pair:
                    pnt1 = pnts1[p[0]].astype(np.int32)
                    self.kp_position1[p[1]] = [pnt1[0], pnt1[1]]
                    self.time_kp_3_displacement_mm[p[1], 0:1] = (self.kp_position1[p[1],
                                                                 0:1] - self.init_kp_2_position_pixel[p[1],
                                                                        0:1]) * self.static_mm_per_pixel
                    self.time_kp_3_displacement_mm[p[1], 2] = self.time_field_depth_mm[pnt1[1], pnt1[0]]
            # 关键点力场
            t3 = time.time()
            self.time_kp_3_force_N = self.time_kp_3_displacement_mm.copy() * 5.0  # todo 5.0比例系数需要根据校准来确定
        t4 = time.time()
        # print('finger_core_alg.run cost time:%.3fs'%(t2-t1))
        # print(self.time_kp_3_force_N.dtype)
        return

    def poisson_reconstruct(self, normals_image, denom):  # grady, gradx, boundarysrc
        # do poisson reconstruction
        # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
        # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf
        gradx = normals_image[:, :, 0] / normals_image[:, :, 2]
        grady = normals_image[:, :, 1] / normals_image[:, :, 2]
        f = np.zeros(gradx.shape)
        f[:-1, 1:] += gradx[:-1, 1:] - gradx[:-1, :-1]
        f[1:, :-1] += grady[1:, :-1] - grady[:-1, :-1]
        f = f[1:-1, 1:-1]
        tt = fftpack.dst(f, norm='ortho')  # Discrete Sine Transform
        fsin = fftpack.dst(tt.T, norm='ortho').T
        f = fsin / denom  # Eigenvalues
        tt = fftpack.idst(f, norm='ortho')  # Inverse Discrete Sine Transform
        f = fftpack.idst(tt.T, norm='ortho').T
        ret = np.zeros(gradx.shape)
        ret[1:-1, 1:-1] = f
        return ret

    def poisson_reconstruct2(self, normals_image, denom):
        gradx = normals_image[:, :, 0] / normals_image[:, :, 2]
        grady = normals_image[:, :, 1] / normals_image[:, :, 2]
        f = np.zeros(gradx.shape)
        f[:-1, 1:] += gradx[:-1, 1:] - gradx[:-1, :-1]
        f[1:, :-1] += grady[1:, :-1] - grady[:-1, :-1]
        f = f[1:-1, 1:-1]
        f = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        tt = fftpack.dst(f, norm='ortho')  # Discrete Sine Transform
        fsin = fftpack.dst(tt.T, norm='ortho').T
        f = fsin / denom  # [::2,::2]# Eigenvalues
        tt = fftpack.idst(f, norm='ortho')  # Inverse Discrete Sine Transform
        f = fftpack.idst(tt.T, norm='ortho').T
        f = cv2.resize(f, (f.shape[1] * 2, f.shape[0] * 2), interpolation=cv2.INTER_LINEAR) * 4
        ret = np.zeros(gradx.shape)
        ret[1:-1, 1:-1] = f
        return ret


if __name__ == '__main__':
    print(0)
    exit()
