# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" evaluation """


import numpy as np
from shapely.geometry import Polygon




def calculate_eao(dataset_name, all_failures, all_overlaps, gt_traj_length, skipping=5):
    '''
    input:dataset name
    all_failures: type is list , index of failure
    all_overlaps: type is  list , length of list is the length of all_failures
    gt_traj_length: type is list , length of list is the length of all_failures
    skipping：number of skipping per failing
    '''
    if dataset_name == "VOT2016":

        low = 108
        high = 371

    elif dataset_name == "VOT2015":
        low = 108
        high = 371

    fragment_num = sum([len(x)+1 for x in all_failures])
    max_len = max([len(x) for x in all_overlaps])
    tags = [1] * max_len
    seq_weight = 1 / (1 + 1e-10) # division by zero

    eao = {}

    # prepare segments
    fweights = np.ones(fragment_num, dtype=np.float32) * np.nan
    fragments = np.ones((fragment_num, max_len), dtype=np.float32) * np.nan
    seg_counter = 0
    for traj_len, failures, overlaps in zip(gt_traj_length, all_failures, all_overlaps):
        if failures:
            points = [x+skipping for x in failures if
                      x+skipping <= len(overlaps)]
            points.insert(0, 0)
            for i, _ in enumerate(points):
                if i != len(points) - 1:
                    fragment = np.array(overlaps[points[i]:points[i+1]+1], dtype=np.float32)
                    fragments[seg_counter, :] = 0
                else:
                    fragment = np.array(overlaps[points[i]:], dtype=np.float32)
                fragment[np.isnan(fragment)] = 0
                fragments[seg_counter, :len(fragment)] = fragment
                if i != len(points) - 1:
                    tag_value = tags[points[i]:points[i+1]+1]
                    w = sum(tag_value) / (points[i+1] - points[i]+1)
                    fweights[seg_counter] = seq_weight * w
                else:
                    tag_value = tags[points[i]:len(overlaps)]
                    w = sum(tag_value) / (traj_len - points[i]+1e-16)
                    fweights[seg_counter] = seq_weight * w
                seg_counter += 1
        else:
            # no failure
            max_idx = min(len(overlaps), max_len)
            fragments[seg_counter, :max_idx] = overlaps[:max_idx]
            tag_value = tags[0: max_idx]
            w = sum(tag_value) / max_idx
            fweights[seg_counter] = seq_weight * w
            seg_counter += 1

    expected_overlaps = calculate_expected_overlap(fragments, fweights)
    print(len(expected_overlaps))
    # calculate eao
    weight = np.zeros((len(expected_overlaps)))
    weight[low-1:high-1+1] = 1
    expected_overlaps = np.array(expected_overlaps, dtype=np.float32)
    is_valid = np.logical_not(np.isnan(expected_overlaps))
    eao_ = np.sum(expected_overlaps[is_valid] * weight[is_valid]) / np.sum(weight[is_valid])
    eao = eao_
    return eao

def calculate_expected_overlap(fragments, fweights):
    """ compute expected iou """
    max_len = fragments.shape[1]
    expected_overlaps = np.zeros((max_len), np.float32)
    expected_overlaps[0] = 1

    # TODO Speed Up
    for i in range(1, max_len):
        mask = np.logical_not(np.isnan(fragments[:, i]))
        if np.any(mask):
            fragment = fragments[mask, 1:i+1]
            seq_mean = np.sum(fragment, 1) / fragment.shape[1]
            expected_overlaps[i] = np.sum(seq_mean *
                                          fweights[mask]) / np.sum(fweights[mask])
    return expected_overlaps

def calculate_accuracy_failures(pred_trajectory, gt_trajectory, \
                                bound=None):
    '''
    args:
    pred_trajectory:list of bbox
    gt_trajectory: list of bbox ,shape == pred_trajectory
    bound :w and h of img
    return :
    overlaps:list ,iou value in pred_trajectory
    acc : mean iou value
    failures: failures point in pred_trajectory
    num_failures: number of failres
    '''


    overlaps = []
    failures = []

    for i, pred_traj in enumerate(pred_trajectory):
        if len(pred_traj) == 1:

            if pred_trajectory[i][0] == 2:
                failures.append(i)
            overlaps.append(float("nan"))

        else:
            if bound is not None:
                poly_img = Polygon(np.array([[0, 0],\
                                    [0, bound[1]],\
                                    [bound[0], bound[1]],\
                                    [bound[0], 0]])).convex_hull


            if len(gt_trajectory[i]) == 8:

                poly_pred = Polygon(np.array([[pred_trajectory[i][0], pred_trajectory[i][1]], \
                                              [pred_trajectory[i][2], pred_trajectory[i][1]], \
                                              [pred_trajectory[i][2], pred_trajectory[i][3]], \
                                              [pred_trajectory[i][0], pred_trajectory[i][3]] \
                                              ])).convex_hull
                poly_gt = Polygon(np.array(gt_trajectory[i]).reshape(4, 2)).convex_hull
                if bound is not None:
                    gt_inter_img = poly_gt.intersection(poly_img)
                    pred_inter_img = poly_pred.intersection(poly_img)
                    inter_area = gt_inter_img.intersection(pred_inter_img).area
                    overlap = inter_area /(gt_inter_img.area + pred_inter_img.area - inter_area)
                else:
                    inter_area = poly_gt.intersection(poly_pred).area
                    overlap = inter_area / (poly_gt.area + poly_pred.area - inter_area)
            elif len(gt_trajectory[i]) == 4:

                overlap = iou(np.array(pred_trajectory[i]).reshape(-1, 4), np.array(gt_trajectory[i]).reshape(-1, 4))
            overlaps.append(overlap)
    acc = 0
    num_failures = len(failures)
    if overlaps:
        acc = np.nanmean(overlaps)
    return acc, overlaps, failures, num_failures

def judge_failures(pred_bbox, gt_bbox, threshold=0):
    """" judge whether to fail or not """
    if len(gt_bbox) == 4:
        if iou(np.array(pred_bbox).reshape(-1, 4), np.array(gt_bbox).reshape(-1, 4)) > threshold:
            return False
    else:
        poly_pred = Polygon(np.array([[pred_bbox[0], pred_bbox[1]], \
                                      [pred_bbox[2], pred_bbox[1]], \
                                      [pred_bbox[2], pred_bbox[3]], \
                                      [pred_bbox[0], pred_bbox[3]] \
                                      ])).convex_hull
        poly_gt = Polygon(np.array(gt_bbox).reshape(4, 2)).convex_hull
        inter_area = poly_gt.intersection(poly_pred).area
        overlap = inter_area / (poly_gt.area + poly_pred.area - inter_area)
        if overlap > threshold:
            return False
    return True

def iou(box1, box2):
    """ compute iou """
    box1, box2 = box1.copy(), box2.copy()
    N = box1.shape[0]
    K = box2.shape[0]
    box1 = np.array(box1.reshape((N, 1, 4)))+np.zeros((1, K, 4))#box1=[N,K,4]
    box2 = np.array(box2.reshape((1, K, 4)))+np.zeros((N, 1, 4))#box1=[N,K,4]
    x_max = np.max(np.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
    x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
    y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
    y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
    tb = x_min-x_max
    lr = y_min-y_max
    tb[np.where(tb < 0)] = 0
    lr[np.where(lr < 0)] = 0
    over_square = tb*lr
    all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (box2[:, :, 2] - \
                                            box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square
    return over_square / all_square
