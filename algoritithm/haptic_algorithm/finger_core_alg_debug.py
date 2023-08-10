import sys, os

sys.path.append(os.path.abspath(f'{os.path.dirname(__file__)}/../src/core'))
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from finger_alg import finger_core_alg
# import vtkmodules.all as vtk


# def draw_vtk(z, static_mm_per_pixel):
#     z = z / static_mm_per_pixel
#     h = z.shape[0]
#     w = z.shape[1]
#     x = np.arange(0, w, 1)
#     y = np.arange(0, h, 1)
#     x, y = np.meshgrid(x, y)
#     # print(w,h,x,y,z)
#     x = x.reshape([w * h, 1])
#     y = y.reshape([w * h, 1])
#     z = z.reshape([w * h, 1])
#     xyz = np.hstack((x, y))
#     xyz = np.hstack((xyz, z))
#
#     # print(w,h,x.shape,y.shape,z.shape,xyz.shape)
#     vs = xyz.astype(np.float32)
#     faces = np.zeros(((w - 1) * (h - 1), 4))
#     k = 0
#     for j in range(h - 1):
#         for i in range(w - 1):
#             faces[k] = np.array([j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i])
#             k += 1
#     faces = faces.astype(np.int32)
#     # print(vs.shape,faces.shape,faces)
#
#     # exit(0)
#
#     # # 0.构造数据
#     # # vs = np.array([[-1, 1, -1],    # 顶点坐标
#     # #                [-1, 0, 0],
#     # #                [-1, -1, -1],
#     # #                [0, 0.5, 0],
#     # #                [0, -0.5, 0],
#     # #                [0.5, 0, -1]], dtype=np.float32)  # 0.5 -0.5
#     # vs = np.array([[0, 0, 0],    # 顶点坐标
#     #             [1, 0, 0],
#     #             [1, 1, 0],
#     #             [0, 1, 1]], dtype=np.float32)  # 0.5 -0.5
#     # # faces = np.array([[4, 1, 3], [4, 1, 2], [0, 3, 1], [3, 5, 4]], dtype=np.int16)  # 三角网格
#     # faces = np.array([[0, 1, 2, 3]], dtype=np.int16)  # 三角网格
#     # # c = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255      # 颜色
#     # # c = np.array([[0.5, 0.5, 0.5]]) * 255      # 颜色
#
#     # 1.vtk创建PolyData
#     points = vtk.vtkPoints()  # 顶点
#     for v in vs:
#         points.InsertNextPoint(v)
#     polys = vtk.vtkCellArray()  # 三角
#     for f in faces:
#         polys.InsertNextCell(4, f)
#     # cellColor = vtk.vtkUnsignedCharArray()    # 存储颜色
#     # cellColor.SetNumberOfComponents(3)        # RGB三通道
#     # for tmp in c:
#     #     cellColor.InsertNextTuple(tmp)
#     cube = vtk.vtkPolyData()  # 分别添加到PolyData
#     cube.SetPoints(points)
#     cube.SetPolys(polys)
#     # cube.GetCellData().SetScalars(cellColor)
#
#     # # 1.5细分
#     # l = vtk.vtkLinearSubdivisionFilter()      # 先linear
#     # l.SetInputData(cube)
#     # l.SetNumberOfSubdivisions(1)
#     # l.Update()
#
#     # loop = vtk.vtkLoopSubdivisionFilter()     # 后loop
#     # loop.SetInputConnection(l.GetOutputPort())
#     # loop.SetNumberOfSubdivisions(5)
#     # loop.Update()
#
#     # 2.创建Mapper
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetColorModeToDefault()
#     mapper.SetInputData(cube)  # 离散网格
#     # mapper.SetInputConnection(loop.GetOutputPort())    # 细分之后的 近似曲面
#
#     # cylinder = vtk.vtkCylinderSource()
#     # cylinder.SetHeight(3.0)
#     # cylinder.SetRadius(1.0)
#     # cylinder.SetResolution(360)
#     # mapper = vtk.vtkPolyDataMapper()
#     # mapper.SetColorModeToDefault()
#     # mapper.SetInputConnection(cylinder.GetOutputPort())    # 细分之后的 近似曲面
#
#     # 3.创建Actor
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     # actor.GetProperty().SetEdgeColor(0, 0, 0)
#     # actor.GetProperty().SetEdgeVisibility(1)    # 显示边
#     # prop = vtk.vtkProperty()
#     # prop.SetColor(0.6,0.96,1)
#     # actor.SetProperty(prop)
#     # bmpReader = vtk.vtkBMPReader()
#     # bmpReader.SetFileName("sky.bmp")
#     # texture = vtk.vtkTexture()
#     # texture.SetInputConnection(bmpReader.GetOutputPort())
#     # texture.InterpolateOn()
#     # actor.SetTexture(texture)
#
#     # 4.创建Renderer
#     renderer = vtk.vtkRenderer()
#     renderer.SetBackground(1, 1, 1)  # 背景白色
#     renderer.AddActor(actor)  # 将actor加入
#     renderer.ResetCamera()  # 调整显示
#
#     # 5.渲染窗口
#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(renderer)
#     renWin.SetSize(1200, 900)
#     renWin.Render()
#
#     # 6.交互
#     renWinInteractor = vtk.vtkRenderWindowInteractor()
#     renWinInteractor.SetRenderWindow(renWin)
#     renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
#     renWinInteractor.Start()
#
#
# def cropROI(img, rect):
#     return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
#
#
# def plot3DVectors2(kp_position0, kp_displacement, resultant_force, force_location, scale=0.005, title='vec3d',
#                    bWait=True):
#     ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')
#     # clear last plot
#     ax.clear()
#
#     # set the axes properties
#     ax.set_xlim3d(0, 20)
#     ax.set_ylim3d(0, 20)
#     ax.set_zlim3d(0, 2)
#
#     # x0, y0, z0, x1, y1, z1, vx, vy, vz
#     xyz0 = kp_position0
#     xyz1 = kp_position0 + kp_displacement[:, :2]
#     vec = kp_displacement
#
#     # ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')
#     x, y, z = xyz0[:, 0], xyz0[:, 1], np.zeros(len(xyz0))
#     x1, y1, z1 = xyz1[:, 0], xyz1[:, 1], kp_displacement[:, 2]
#     u, v, w = vec[:, 0], vec[:, 1], vec[:, 2]
#     ax.scatter(x, y, z, marker='o')
#     ax.scatter(x1, y1, z1, marker='^')
#     force = resultant_force[:3] * scale
#     ax.scatter(force_location[0], force_location[1], force_location[2], marker='o', s=200)
#     ax.scatter(force_location[0] + force[0], force_location[1] + force[1], force_location[2] + force[2], marker='^',
#                s=200)
#     # scale = 0.05
#     q = ax.quiver(force_location[0], force_location[1], force_location[2], force[0], force[1], force[2],
#                   arrow_length_ratio=0.1, linewidths=3, colors=[1, 0, 0])
#     q = ax.quiver(x, y, z, u, v, w)  # , arrow_length_ratio=0.1, length=0.5, normalize=True)
#     ax.set_title(title)
#     # plt.savefig('plot3DVectors.png')  # comment this line(by JadeCong)
#     # if bWait:
#     #     plt.show(block=False)
#     #     plt.pause(0.0001)
#     plt.show()


# def draw_cv2(finger_core_run, img_color1):
#     img_show = cropROI(img_color1, finger_core_run.roi)  # get region of interest
#     # img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
#     # img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
#     # for kp in zip(finger_core_run.init_kp_2_position_pixel, finger_core_run.kp_position1):
#     #     cv2.circle(img_show, (np.int32(kp[0][0]), np.int32(kp[0][1])), 4, (255, 0, 0),
#     #                2)  # changed p to tuple(p) (by JadeCong)
#     #     cv2.rectangle(img_show, (np.int32(kp[1][0]) - 4, np.int32(kp[1][1]) - 4),
#     #                   (np.int32(kp[1][0]) + 4, np.int32(kp[1][1]) + 4), (0, 255, 0), 1)
#     #     cv2.line(img_show, (np.int32(kp[0][0]), np.int32(kp[0][1])), (np.int32(kp[1][0]), np.int32(kp[1][1])),
#     #              (0, 0, 255), 1)
#     # img_show = cv2.resize(img_show, (img_show.shape[1]*2,img_show.shape[0]*2), interpolation= cv2.INTER_LINEAR)
#     # cv2.imshow('img_color1', img_color1)
#     # cv2.imshow('imshow', img_show)
#     cv2.imshow('displacement', finger_core_run.time_field_depth_mm)
#     ret = cv2.waitKey()
#     cv2.destroyAllWindows()


# def draw_matplotlib(finger_core_run):
#     location = finger_core_run.calc_force_location(finger_core_run.resultant_force)
#     location[0] += finger_core_run.roi[2] * finger_core_run.pixel_to_mm_scale * 0.5
#     location[1] += finger_core_run.roi[3] * finger_core_run.pixel_to_mm_scale * 0.5
#     plot3DVectors2(finger_core_run.kp_position0 * finger_core_run.pixel_to_mm_scale, finger_core_run.kp_displacement,
#                    finger_core_run.resultant_force, location, title='vec3d', bWait=True)


def fit_ellipse(depth_img, depth_threshold=0.4):
    x_coord, y_coord = np.where(depth_img >= depth_threshold)
    points = list(zip(y_coord.tolist(), x_coord.tolist()))
    coord_cotton_points = np.array(points)
    ellipse = cv2.fitEllipse(coord_cotton_points)
    ellipse = cv2.ellipse(depth_img, ellipse, (255, 255, 0), 2)
    return ellipse


def fit_line(depth_img, depth_threshold=0.4):
    x_coord, y_coord = np.where(depth_img >= depth_threshold)
    points = list(zip(x_coord.tolist(), y_coord.tolist()))
    coord_cotton_points = np.array(points)
    print(coord_cotton_points.shape)
    output = cv2.fitLine(coord_cotton_points, distType=cv2.DIST_L2, param=0, reps=1e-2, aeps=1e-2)
    cos_theta, sin_theta, x0, y0 = output
    slope = sin_theta / cos_theta
    intercept = y0 - slope * x0
    return slope, intercept, (x0, y0)


def draw_line(img_color, img_depth, slope, intercept, be_write=False):

    w, h = img_depth.shape[:2]
    y1 = 1
    x1 = int(slope * y1 + intercept)
    y2 = h - 20
    x2 = int(slope * y2 + intercept)

    # img_depth_clr = cv2.applyColorMap(cv2.convertScaleAbs(img_depth * 100, alpha=15), cv2.COLORMAP_JET)

    depth_img_3D = np.stack((img_depth, img_depth, img_depth), 2)
    cv2.line(img_depth, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 255), 2)
    img_show = np.concatenate((img_color, depth_img_3D), axis=1)

    if be_write:
        save_img_dir = r'D:\code\python\nucleic_acid_sampling_alg\algoritithm\haptic_algorithm\result_image_show' \
                       r'\draw_out'
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        img_save_id = os.listdir(save_img_dir).__len__()
        save_img_path = os.path.join(save_img_dir, f'{img_save_id}.png')
        cv2.imwrite(save_img_path, img_show)

    cv2.imshow('img_color', img_color)
    cv2.imshow('img_depth', img_depth)
    # cv2.imshow('image', img_show)


if __name__ == '__main__':
    # data_folder = f"{os.path.abspath(os.path.dirname(__file__))}/data/2022.03.30/"
    # calib_file = f'{data_folder}filtered_2022-03-30T12.15.38.328608.calib'
    # background_file = f'{data_folder}background/capture_background.png'
    # mask_file = f'{data_folder}background/mask.png'
    # image_file = f'{data_folder}ball.6mm/capture_ball_7.png'

    data_folder = f"{os.path.abspath(os.path.dirname(__file__))}\\calibration\\220715-005\\"
    calib_file = f'{data_folder}calibration.txt'
    background_file = f'{data_folder}background.png'
    mask_file = f'{data_folder}mask.png'
    # image_file = f'{data_folder}image011.png'
    image_file = r'D:\code\python\sensor\finger_alg-main (2)\finger_alg-main\test\data\dwn\35.png'

    finger_core_run = finger_core_alg(calib_file, background_file, mask_file, 0)
    cap = cv2.VideoCapture(0)
    # import json
    # with open('camera_params.json', 'r') as f:
    #     camera_setup = json.load(f)
    #     setting = camera_setup.keys()
    #     for k in setting[:42]:
    #         try:
    #             cap.set(k, camera_setup[k])
    #         except TypeError:
    #             cap.set(int(k), camera_setup[k])
    cap.set(3, 1920)  # width=1920
    cap.set(4, 1080)  # height=1080

    depth_threshold = 0.3
    while True:
        # img = cv2.imread(image_file)
        # for _ in range(1):
        ret, img = cap.read()
        assert ret == True, "please check your camera"

        t0 = time.time()
        finger_core_run.calc(img, 0)
        t1 = time.time()
        print(np.max(finger_core_run.time_field_depth_mm))

        depth_info = finger_core_run.time_field_depth_mm
        img_depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_info * 255, alpha=15), cv2.COLORMAP_JET)
        depth_info[depth_info < depth_threshold] = 0
        max_depth = np.max(depth_info)
        cv2.imshow('img_color', img_depth_color)
        if max_depth >= depth_threshold:
            stop_moving_times = 0
            start_time = time.time()
            init_k = 0
            init_b = 0
            try:
                init_k, init_b, _ = fit_line(depth_info)
                fit_ellipse(depth_info)
            except cv2.error:
                continue
            dist_threshold = 7

            while True:
                t2 = time.time()
                ret, img = cap.read()
                finger_core_run.calc(img, 0)
                depth_img = finger_core_run.time_field_depth_mm
                t3 = time.time()
                print('finger_core_alg.run cost time:%.3fs' % (t3 - t2))

                img_clr = np.copy(finger_core_run.img)
                img_depth = np.copy(depth_img)
                try:
                    t4 = time.time()
                    k, b, _ = fit_line(depth_img)
                    # fit_ellipse(depth_info)
                    t5 = time.time()
                    print('fit line cost time:%.3fs' % (t5 - t4))
                except cv2.error:
                    break
                dist_k_b = 10 * abs(k - init_k) + abs(b - init_b)

                if dist_k_b > dist_threshold:
                    cv2.putText(img_clr, "moving", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    draw_line(img_clr, img_depth, k, b)
                    c = cv2.waitKey(5)
                    if c == 27:
                        cv2.destroyAllWindows()
                        break
                else:
                    stop_moving_times += 1
                    cv2.putText(img_clr, "no moving", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    draw_line(img_clr, img_depth, k, b)
                    c = cv2.waitKey(5)
                    if c == 27:
                        cv2.destroyAllWindows()
                        break
                init_k = k
                init_b = b

        # cv2.imwrite(f'./data/depth_img/depth_img{i}.png', depth_info)
        # i += 1

        cv2.putText(finger_core_run.img, "no cotton", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow('img_depth', depth_info)
        cv2.imshow('img_color', finger_core_run.img)
        c = cv2.waitKey(10)
        if c == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    # draw_matplotlib(finger_core_run)
    # draw_vtk(finger_core_run.time_field_depth_mm, finger_core_run.static_mm_per_pixel)
