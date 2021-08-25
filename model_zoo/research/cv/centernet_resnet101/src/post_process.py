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
"""
Post-process functions after decoding
"""
import numpy as np
from .image import get_affine_transform, affine_transform, transform_preds
from .visual import coco_box_to_bbox


def post_process(dets, meta, scale, num_classes):
    """rescale detection to original scale"""
    c, s, h, w = meta['c'], meta['s'], meta['out_height'], meta['out_width']
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c, s, (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c, s, (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)

    for j in range(1, num_classes + 1):
        ret[0][j] = np.array(ret[0][j], dtype=np.float32).reshape(-1, 5)
        ret[0][j][:, :4] /= scale
    return ret[0]


def merge_outputs(detections, num_classes, SOFT_NMS=True):
    """merge detections together by nms"""
    results = {}
    max_per_image = 100
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)
        if SOFT_NMS:
            try:
                from nms import soft_nms
            except ImportError:
                print('NMS not installed! Do \n cd $CenterNet_ROOT/scripts/ \n'
                      'and see run_standalone_eval.sh for more details to install it\n')
            soft_nms(results[j], Nt=0.5, threshold=0.001, method=2)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def convert_eval_format(detections, img_id, _valid_ids):
    """convert detection to annotation json format"""
    pred_anno = {"images": [], "annotations": []}
    for cls_ind in detections:
        class_id = _valid_ids[cls_ind - 1]
        for det in detections[cls_ind]:
            score = det[4]
            bbox = det[0:4]
            bbox[2:4] = det[2:4] - det[0:2]
            bbox = list(map(to_float, bbox))

            pred = {
                "image_id": int(img_id),
                "category_id": int(class_id),
                "bbox": bbox,
                "score": to_float(score),
            }
            pred_anno["annotations"].append(pred)
    if pred_anno["annotations"]:
        pred_anno["images"].append({"id": int(img_id)})
    return pred_anno


def to_float(x):
    """format float data"""
    return float("{:.2f}".format(x))


def resize_detection(detection, pred, gt):
    """resize object annotation info"""
    height, width = gt[0], gt[1]
    c = np.array([pred[1] / 2., pred[0] / 2.], dtype=np.float32)
    s = max(pred[0], pred[1]) * 1.0
    trans_output = get_affine_transform(c, s, 0, [width, height])

    anns = detection["annotations"]
    num_objects = len(anns)
    resized_detection = {"images": detection["images"], "annotations": []}
    for i in range(num_objects):
        ann = anns[i]
        bbox = coco_box_to_bbox(ann['bbox'])
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        bbox = [bbox[0], bbox[1], w, h]
        ann["bbox"] = list(map(to_float, bbox))
        resized_detection["annotations"].append(ann)
    return resize_detection
