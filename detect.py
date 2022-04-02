import argparse
import os
import shutil
from pathlib import Path

import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, scale_labels, write_txt,
    draw_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, time_synchronized


def detect():
    '''
    input: save_img_flag
    output(result):
    '''
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    if half:
        model.half()  # to FP16

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    for path, img, im0s, vid_cap in dataset:
        print(img.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, without_iouthres=False)
        t2 = time_synchronized()

        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()

                # Print results    det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])

                for *rbox, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        classname = '%s' % names[int(cls)]
                        conf_str = '%.3f' % conf
                        write_txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt'))
                        draw_box(rbox, im0, classname, label=label, color=colors[int(cls)], line_thickness=1,
                                 pi_format=False)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            cv2.imwrite(save_path, im0)
    print('   Results saved to %s' % Path(out))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/cornell.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='DataSet/cornell/images', help='source')
    parser.add_argument('--output', type=str, default='DataSet/cornell/detection', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
