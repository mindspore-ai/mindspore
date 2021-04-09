# Copyright 2020 Huawei Technologies Co., Ltd
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
"""coco eval for maskrcnn"""
import json
import numpy as np
import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from src.config import config

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000), single_result=False):
    """coco eval for maskrcnn"""
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                cocoEval = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)

                cocoEval.params.imgIds = [id_i]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                res_dict.update({coco.imgs[id_i]['file_name']: cocoEval.stats[1]})

        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        summary_metrics = {
            'Precision/mAP': cocoEval.stats[0],
            'Precision/mAP@.50IOU': cocoEval.stats[1],
            'Precision/mAP@.75IOU': cocoEval.stats[2],
            'Precision/mAP (small)': cocoEval.stats[3],
            'Precision/mAP (medium)': cocoEval.stats[4],
            'Precision/mAP (large)': cocoEval.stats[5],
            'Recall/AR@1': cocoEval.stats[6],
            'Recall/AR@10': cocoEval.stats[7],
            'Recall/AR@100': cocoEval.stats[8],
            'Recall/AR@100 (small)': cocoEval.stats[9],
            'Recall/AR@100 (medium)': cocoEval.stats[10],
            'Recall/AR@100 (large)': cocoEval.stats[11],
        }

    return summary_metrics


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
        ]

def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]

    return result

def proposal2json(dataset, results):
    """convert proposal to json mode"""
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = dataset.get_dataset_size()*2
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results

def det2json(dataset, results):
    """convert det to json mode"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = len(img_ids)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results): break
        result = results[idx]
        for label, result_label in enumerate(result):
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results

def segm2json(dataset, results):
    """convert segm to json mode"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    bbox_json_results = []
    segm_json_results = []

    dataset_len = len(img_ids)
    assert dataset_len == len(results)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results): break
        det, seg = results[idx]
        for label, det_label in enumerate(det):
            bboxes = det_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                bbox_json_results.append(data)

            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results

def results2json(dataset, results, out_file):
    """convert result convert to json mode"""
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files

def get_seg_masks(mask_pred, det_bboxes, det_labels, img_meta, rescale, num_classes):
    """Get segmentation masks from mask_pred and bboxes"""
    mask_pred = mask_pred.astype(np.float32)

    cls_segms = [[] for _ in range(num_classes - 1)]
    bboxes = det_bboxes[:, :4]
    labels = det_labels + 1

    ori_shape = img_meta[:2].astype(np.int32)
    scale_factor = img_meta[2:].astype(np.int32)

    if rescale:
        img_h, img_w = ori_shape[:2]
    else:
        img_h = np.round(ori_shape[0] * scale_factor[0]).astype(np.int32)
        img_w = np.round(ori_shape[1] * scale_factor[1]).astype(np.int32)

    for i in range(bboxes.shape[0]):
        bbox = (bboxes[i, :] / 1.0).astype(np.int32)
        label = labels[i]
        w = max(bbox[2] - bbox[0] + 1, 1)
        h = max(bbox[3] - bbox[1] + 1, 1)
        w = min(w, img_w - bbox[0])
        h = min(h, img_h - bbox[1])
        if w <= 0 or h <= 0:
            print("there is invalid proposal bbox, index={} bbox={} w={} h={}".format(i, bbox, w, h))
            w = max(w, 1)
            h = max(h, 1)
        mask_pred_ = mask_pred[i, :, :]
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bbox_mask = mmcv.imresize(mask_pred_, (w, h))
        bbox_mask = (bbox_mask > config.mask_thr_binary).astype(np.uint8)
        im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

        rle = maskUtils.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        cls_segms[label - 1].append(rle)

    return cls_segms
