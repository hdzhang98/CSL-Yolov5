import glob
import logging
import math
import os
import random
import time
import shapely
import shapely.geometry
import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import pdb


torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})


def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def check_img_size(img_size, s=32):
    '''
    Verify img_size is a multiple of stride s
    return new_size
    '''
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_file(file):
    '''
    Search for file if not found
    '''
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x):
    '''
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    '''
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_labels(img1_shape, labels, img0_shape, ratio_pad=None):
    '''
    Rescale coords (xywh) from img1_shape to img0_shape
    @return:
            scaled_labels : (num ,[ x y longside shortside Θ])
    '''

    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    scaled_labels = []
    for i, label in enumerate(labels):
        rect = ((label[0], label[1]), (label[2], label[3]), label[4])
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)

        poly[:, 0] -= pad[0]   # x padding
        poly[:, 1] -= pad[1]   # y padding
        poly[:, :] /= gain
        clip_poly(poly, img0_shape)

        rect_scale = cv2.minAreaRect(np.float32(poly))

        c_x = rect_scale[0][0]
        c_y = rect_scale[0][1]
        w = rect_scale[1][0]
        h = rect_scale[1][1]
        theta = rect_scale[-1]  # Range for angle is [-90，0)

        label = np.array(point2longsideformat(c_x, c_y, w, h, theta))

        label[-1] = int(label[-1] + 180.5)
        if label[-1] == 180:
            label[-1] = 179
        scaled_labels.append(label)

    return torch.from_numpy(np.array(scaled_labels))


def clip_poly(poly, img_shape):
    '''
    Clip bounding [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] bounding boxes to image shape (height, width)
    '''
    poly[:, 0].clip(0, img_shape[1])  # x
    poly[:, 1].clip(0, img_shape[0])  # y


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):
    '''
    return positive, negative label smoothing BCE targets
    '''
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def gaussian_label(label, num_class, u=0, sig=6.0):
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    return np.concatenate([y_sig[math.ceil(num_class / 2) - int(label.item()):],
                           y_sig[:math.ceil(num_class / 2) - int(label.item())]], axis=0)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    '''
    Performs Non-Maximum Suppression (NMS) on inference results；
    @param prediction:  size=(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    @param conf_thres:
    @param iou_thres:
    @param merge:
    @param classes:
    @param agnostic:
    @return:
            detections with shape: (batch_size, num_nms_boxes, [])
    '''

    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    nc = prediction[0].shape[1] - 5  # number of classes
    class_index = nc + 5
    # xc : (batch_size, num_boxes)
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)


    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # x -> (num_confthres_boxes, [xywh,score,num_classes,Θ])
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)
        # box.size = (num_confthres_boxes, [xywhθ])  θ ∈ [-pi/2, pi/2) length=180
        box = torch.cat((x[:, :4], (angle_index - 90) * np.pi / 180), 1)

        # Detections matrix nx7 (xywhθ, conf, clsid) θ ∈ [-pi/2, pi/2)
        if multi_label:
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            # list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue

        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classesid*4096
        boxes, scores = x[:, :5], x[:, 5]  # boxes[x, y, w, h, θ] θ ∈ [-pi/2, pi/2)
        boxes[:, :4] = boxes[:, :4] + c  # boxes xywh(offset by class)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def rotate_nms(dets, scores, thresh):
    x1 = np.min(dets[:, 0::2], axis=1)  # (num, 1)
    y1 = np.min(dets[:, 1::2], axis=1)  # (num, 1)
    x2 = np.max(dets[:, 0::2], axis=1)  # (num, 1)
    y2 = np.max(dets[:, 1::2], axis=1)  # (num, 1)
    polys = []
    for i in range(len(dets)):
        polys.append(dets[i])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter
        h_inds = np.where(hbb_inter > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polygon_iou(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        idx = np.where(hbb_ovr <= thresh)[0]
        order = order[idx + 1]
    return keep


def polygon_iou(boxes1, boxes2):
    polygon1 = shapely.geometry.Polygon(boxes1.reshape(4, 2)).convex_hull
    polygon2 = shapely.geometry.Polygon(boxes2.reshape(4, 2)).convex_hull
    if polygon1.intersects(polygon2):
        inter = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou = inter / union
    else:
        iou = 0
    return iou


def rotate_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None, agnostic=False, without_iouthres=False):
    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,num_angles])
    nc = prediction[0].shape[1] - 5 - 180  # number of classes = no - 5 -180
    class_index = nc + 5
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  #
    # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes, [xywh, score, num_classes, num_angles])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]
        angle = x[:, class_index:]
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)
        box = torch.cat((x[:, :4], angle_index), 1)
        if multi_label:
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if without_iouthres:
            output[xi] = x
            continue
        if classes:
            # list x：(num_confthres_boxes, [xywhθ,conf,classid])
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]
        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        boxes_xy, box_whthetas, scores = x[:, :2] + c, x[:, 2:5], x[:, 5]
        rects = []
        for i, box_xy in enumerate(boxes_xy):
            rect = longsideformat2point(box_xy[0], box_xy[1], box_whthetas[i][0], box_whthetas[i][1], box_whthetas[i][2])
            rects.append(rect)
        i = np.array(rotate_nms(np.array(rects), np.array(scores.cpu()), iou_thres))
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def longsideformat2point(x_c, y_c, longside, shortside, theta_longside):
    theta = (theta_longside / 180) * math.pi
    origin_theta = math.atan(shortside / longside)
    r = math.sqrt(longside * longside + shortside * shortside) / 2
    px1 = int(x_c + r * math.cos(origin_theta + theta))
    py1 = int(y_c - r * math.sin(origin_theta + theta))
    px2 = int(x_c - r * math.cos(origin_theta - theta))
    py2 = int(y_c - r * math.sin(origin_theta - theta))
    px3 = int(x_c - r * math.cos(origin_theta + theta))
    py3 = int(y_c + r * math.sin(origin_theta + theta))
    px4 = int(x_c + r * math.cos(origin_theta - theta))
    py4 = int(y_c + r * math.sin(origin_theta - theta))
    poly = np.double([px1, py1, px2, py2, px3, py3, px4, py4])
    return poly


def point2longsideformat(x_c, y_c, width, height, theta):
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if width != max(width, height):
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:
        longside = width
        shortside = height
        theta_longside = theta

    return x_c, y_c, longside, shortside, theta_longside


def draw_box(rbox, img, color=None, label=None, line_thickness=None, pi_format=True):
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # rbox = np.array(x)
    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]
    # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
    rect = ((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4])
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.float32(cv2.boxPoints(rect))
    poly = np.int0(poly)
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=color, thickness=2*tl)
    c1 = (int(rbox[0]), int(rbox[1]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3


def write_txt(rbox, classname, conf, img_name, out_path, pi_format=False):
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]
    poly = longsideformat2point(rbox[0], rbox[1], rbox[2], rbox[3], rbox[4])
    lines = img_name + ' ' + conf + ' ' + ' '.join(list(map(str, poly))) + ' ' + classname
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # make new output folder

    with open(str(out_path + '/' + img_name) + '.txt', 'a') as f:
        f.writelines(lines + '\n')


