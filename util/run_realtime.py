# coding = utf-8
# load yolox model

import argparse
import glob
import os
import cv2
import numpy as np
import time
from pygame import mixer
import torch

from algoritithm.haptic_algorithm.finger_alg import finger_core_alg
from algoritithm.vision_algorithm.demo import Predictor, get_model
from algoritithm.vision_algorithm.exp import get_exp
from algoritithm.vision_algorithm.voc_classes import VOC_CLASSES


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="result_image_show", help="demo type, eg. result_image_show, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='exps/example/yolox_voc/yolox_voc_s.py')
    parser.add_argument("-n", "--name", type=str, default='yolox_s', help="model name")

    parser.add_argument(
        "--path", default="./assets", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        # action="store_true",
        type=bool,
        default=False,
        help="whether to save the inference result of result_image_show/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./checkpoints/yolox_voc_s/latest_ckpt.pth', type=str,
                        help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.8, type=float, help="test conf")
    parser.add_argument("--nms", default=0.2, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    parser.add_argument(
        '--calib',
        default=r'D:\code\python\nucleic_acid_sampling_alg\algoritithm\haptic_algorithm\calibration\220715-005',
        type=str,
        help='camera calibration files. Include mask, background, calibration'
    )
    return parser


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
    return img_color, img_depth


def init_yolox(args):
    # exp = yolox_voc_s.Exp()
    exp = get_exp(args.exp_file, args.name)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = get_model(exp)
    if not args.trt:
        ckpt_file = args.ckpt
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        print("loaded checkpoint done.")
    voc_classes = VOC_CLASSES
    trt_file = None
    decoder = None
    predictor = Predictor(
        model, exp, voc_classes, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    return predictor


def init_finger(args):

    data_folder = args.calib
    if not args.calib:
        data_folder = r'D:\code\python\nucleic_acid_sampling_alg\algoritithm\haptic_algorithm\calibration\220715-005'

    calib_file = os.path.join(data_folder, 'calibration.txt')
    background_file = os.path.join(data_folder, 'background.png')
    mask_file = os.path.join(data_folder, 'mask.png')
    finger_core_run = finger_core_alg(calib_file, background_file, mask_file, 0)
    return finger_core_run


def box_from_outputs(outputs_ts):
    box = outputs_ts[0, :4]
    return box


def box_list_from_array(out_arr):
    box = [out_arr[0], out_arr[0], out_arr[0], out_arr[0]]
    return box


def find_max_conf_info(inp_ts):
    max_ind = torch.argmax(inp_ts[:, 4])
    inp_max_conf_ts = inp_ts[max_ind].reshape((-1, 7))
    return inp_max_conf_ts


def get_max_conf(output_ts):
    if output_ts is None:
        return None, [None, None]

    cotton_swap_ind = torch.argwhere(output_ts[:, -1] == 0)
    soft_palate_ind = torch.argwhere(output_ts[:, -1] == 1)

    soft_palate_info = None if not soft_palate_ind.shape[0] else output_ts[soft_palate_ind].reshape((-1, 7))
    cotton_swap_info = None if not cotton_swap_ind.shape[0] else output_ts[cotton_swap_ind].reshape((-1, 7))

    soft_palate_max_conf_info = None if soft_palate_info is None else find_max_conf_info(soft_palate_info)
    cotton_swap_max_conf_info = None if cotton_swap_info is None else find_max_conf_info(cotton_swap_info)

    soft_palate_info = soft_palate_max_conf_info
    cotton_swap_info = cotton_swap_max_conf_info
    out_info_ts = None
    if soft_palate_info is not None and cotton_swap_info is not None:
        out_info_ts = torch.concat([soft_palate_info, cotton_swap_max_conf_info], dim=0)

    if soft_palate_info is None:
        out_info_ts = cotton_swap_info

    if cotton_swap_info is None:
        out_info_ts = soft_palate_info

    return out_info_ts, [soft_palate_info, cotton_swap_info]


def cal_depth_from_finger(finger_core_run, img, depth_threshold):
    # calculate depth
    t0 = time.time()
    finger_core_run.calc(img, 0)
    t1 = time.time()
    print('finger_core_alg.run cost time:%.3fs' % (t1 - t0))

    depth_info = finger_core_run.time_field_depth_mm
    depth_info[depth_info < depth_threshold] = 0
    max_depth = np.max(depth_info)
    print(f'max depth : {max_depth}')
    return max_depth, finger_core_run


def show_img(img_finger=None, img_depth=None, img_for_det=None):
    # cv2.namedWindow("detect", 0)
    # cv2.resizeWindow("detect", 640, 640)
    #
    # cv2.namedWindow("finger color", 0)
    # cv2.resizeWindow("finger color", 640, 640)
    #
    # cv2.namedWindow("finger depth", 0)
    # cv2.resizeWindow("finger depth", 640, 640)

    if img_for_det is not None:
        cv2.imshow('detect', img_for_det)
    if img_finger is not None:
        cv2.imshow('finger color', img_finger)
    if img_depth is not None:
        cv2.imshow('finger depth', img_depth)


def be_include(box_small, box_big):
    res0 = box_big[0] < box_small[0]
    res1 = box_big[1] < box_small[1]
    res2 = box_big[2] > box_small[2]
    res3 = box_big[3] > box_small[3]
    if res0 and res1 and res2 and res3:
        return True
    else:
        return False


def main():
    args = make_parser().parse_args()
    detection_prediction = init_yolox(args)
    finger_core_run = init_finger(args)

    sound_file_path = {"begin_sample": glob.glob('data/sound/begin*/*.mp3')[0],
                       "put_cotton_swap": glob.glob('data/sound/put*/*.mp3')[0],
                       "draw_out_cotton_swap": glob.glob('data/sound/draw*/*.mp3')[0],
                       "sample_success": glob.glob('data/sound/sample_success/*.mp3')[0]}

    mixer.init()  # initialzing pyamge mixer
    cap_finger = cv2.VideoCapture(0)
    cap_finger.set(3, 1920)  # width=1920
    cap_finger.set(4, 1080)  # height=1080

    cap_detection = cv2.VideoCapture(1)

    depth_threshold = 0.3
    while True:
        ret_finger, img_finger = cap_finger.read()
        ret_det, img_det = cap_detection.read()
        assert ret_finger and ret_det, "please check your camera"

        # calculate depth
        t0 = time.time()
        finger_core_run.calc(img_finger, 0)
        t1 = time.time()
        print('finger_core_alg.run cost time:%.3fs' % (t1 - t0))

        depth_info = finger_core_run.time_field_depth_mm
        depth_info[depth_info < depth_threshold] = 0
        max_depth = np.max(depth_info)
        print(f'max depth : {max_depth}')

        # 判断是否夹紧棉签
        if max_depth >= depth_threshold:
            mixer.music.stop()

            try:
                init_k, init_b, _ = fit_line(depth_info)
            except cv2.error:
                continue

            dist_threshold = 7
            valid_times = 0
            sample_valid_time = 8

            # calculate valid time
            while valid_times < sample_valid_time:
                t2 = time.time()
                ret, img_finger = cap_finger.read()
                finger_core_run.calc(img_finger, 0)
                depth_img = finger_core_run.time_field_depth_mm
                t3 = time.time()
                print('finger_core_alg.run cost time:%.3fs' % (t3 - t2))

                ret_det, img_det = cap_detection.read()
                outputs, img_info = detection_prediction.inference(img_det)
                max_conf_output, sp_cs_info_list = get_max_conf(outputs[0])
                detect_frame = detection_prediction.visual(max_conf_output, img_info, detection_prediction.confthre)

                img_clr = np.copy(finger_core_run.img)
                img_depth = np.copy(depth_img)
                try:
                    t4 = time.time()
                    k, b, _ = fit_line(depth_img)
                    # fit_ellipse(depth_info)
                    t5 = time.time()
                    print('fit line cost time:%.3fs' % (t5 - t4))
                except cv2.error:
                    continue

                # 计算直线变化程度
                dist_k_b = 10 * abs(k - init_k) + abs(b - init_b)

                # 面前在软腭区域内，棉签拟合直线在动
                sp_cs_info_list.append(max_conf_output)
                if None in sp_cs_info_list:
                    cv2.putText(detect_frame, f"sample_valid_per: {round(valid_times/sample_valid_time, 4) * 100}%", (5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(detect_frame, f"sampling", (5, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                    show_img(finger_core_run.img, finger_core_run.time_field_depth_mm, detect_frame)
                    cv2.waitKey(5)
                    continue

                soft_palate_boxes = box_from_outputs(sp_cs_info_list[0])
                cotton_boxes = box_from_outputs(sp_cs_info_list[1])
                be_in = be_include(cotton_boxes, soft_palate_boxes)
                if dist_k_b > dist_threshold and be_in:
                    valid_times += 1
                    cv2.putText(img_clr, "moving", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(detect_frame, f"sample_valid_per: {round(valid_times / sample_valid_time, 4) * 100}%",
                                (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(detect_frame, f"sampling", (5, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                    img_clr, img_depth = draw_line(img_clr, img_depth, k, b)
                    show_img(img_clr, img_depth, detect_frame)
                    c = cv2.waitKey(5)
                    if c == 27:
                        cv2.destroyAllWindows()
                        break
                else:
                    cv2.putText(img_clr, "no moving", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(detect_frame, f"sample_valid_per: {round(valid_times / sample_valid_time, 4) * 100}%",
                                (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(detect_frame, f"sampling", (5, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    img_clr, img_depth = draw_line(img_clr, img_depth, k, b)
                    show_img(img_clr, img_depth, detect_frame)
                    c = cv2.waitKey(5)
                    if c == 27:
                        cv2.destroyAllWindows()
                        break
                init_k = k
                init_b = b

            if valid_times == sample_valid_time:
                mixer.music.load(sound_file_path['sample_success'])
                mixer.music.play()
                
            if valid_times < sample_valid_time:
                cv2.destroyWindow('detect')

            print('detect whether drawing out cotton swap')
            # detect whether drawing out cotton swap
            _, img_finger_det_draw_out = cap_finger.read()
            max_depth, finger_core_run = cal_depth_from_finger(finger_core_run, img_finger_det_draw_out, depth_threshold)
            while max_depth > depth_threshold and valid_times >= sample_valid_time:
                _, img_finger_det_draw_out = cap_finger.read()
                _, img_det = cap_detection.read()
                max_depth, finger_core_run = cal_depth_from_finger(finger_core_run, img_finger_det_draw_out,
                                                                   depth_threshold)
                cv2.putText(img_det, f"sample_valid_per: {100}%", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 255), 2)
                cv2.putText(img_det, f"sampling success, please draw out cotton swap", (5, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                show_img(img_finger=finger_core_run.img, img_depth=finger_core_run.time_field_depth_mm,
                         img_for_det=img_det)
                c = cv2.waitKey(10)
            cv2.destroyWindow('detect')

        cv2.putText(finger_core_run.img, "no cotton", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        # show_img(finger_core_run.img, finger_core_run.time_field_depth_mm, img_det)
        show_img(finger_core_run.img, finger_core_run.time_field_depth_mm)
        c = cv2.waitKey(10)
        if c == 27:
            cap_finger.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()