"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor

MIT License

Copyright (c) 2018 Vic Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division

import os
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat')) # you own ground_truth name
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):
    """
    Get gt boxes from binary txt file.
    """
    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    """Get preds"""
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = events
    for event in pbar:
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, box = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = box
        boxes[event] = current_event
    return boxes


def norm_score(pred_norm):
    """ norm score
    pred_norm {key: [[x1,y1,x2,y2,s]]}
    """
    max_score = 0
    min_score = 1

    for _, k in pred_norm.items():
        for _, v in k.items():
            if v.size == 0:
                continue
            min_v = np.min(v[:, -1])
            max_v = np.max(v[:, -1])
            max_score = max(max_v, max_score)
            min_score = min(min_v, min_score)

    diff = max_score - min_score
    for _, k in pred_norm.items():
        for _, v in k.items():
            if v.size == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred_eval, gt, ignore, iou_thresh):
    """ single image evaluation
    pred_eval: Nx5
    gt: Nx4
    ignore:
    """
    pred_t = pred_eval.copy()
    gt_t = gt.copy()
    pred_recall = np.zeros(pred_t.shape[0])
    recall_list = np.zeros(gt_t.shape[0])
    proposal_list = np.ones(pred_t.shape[0])

    pred_t[:, 2] = pred_t[:, 2] + pred_t[:, 0]
    pred_t[:, 3] = pred_t[:, 3] + pred_t[:, 1]
    gt_t[:, 2] = gt_t[:, 2] + gt_t[:, 0]
    gt_t[:, 3] = gt_t[:, 3] + gt_t[:, 1]

    overlaps = bbox_overlaps(pred_t[:, :4], gt_t)

    for h in range(pred_t.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    """
    Image pr info
    """
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if r_index.size == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    pr_curve_t = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        pr_curve_t[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        pr_curve_t[i, 1] = pr_curve[i, 1] / count_face
    return pr_curve_t


def voc_ap(rec, prec):
    """
    Voc ap calculation
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred_evaluation, gt_path, iou_thresh=0.4):
    """
    evaluation method.
    """
    print_pred = pred_evaluation
    pred_evaluation = get_preds(pred_evaluation)
    norm_score(pred_evaluation)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = range(event_num)
        error_count = 0
        for i in pbar:
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred_evaluation[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j, _ in enumerate(img_list):
                try:
                    pred_info = pred_list[str(img_list[j][0][0])]
                except KeyError:
                    error_count += 1
                    continue

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                if gt_boxes.size == 0 or pred_info.size == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if keep_index.size != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results = ====================", print_pred)
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='',
                        help='test output, txt contain box positions and scores')
    parser.add_argument('-g', '--gt', default='', help='ground truth path, mat format')
    args = parser.parse_args()

    pred = args.pred
    if os.path.isdir(pred):
        evaluation(pred, args.gt)
    else:
        pass
