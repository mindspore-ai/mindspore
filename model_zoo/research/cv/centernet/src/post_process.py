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
"""
Post-process functions after decoding
"""

import numpy as np
from src.config import dataset_config as config
from .image import get_affine_transform, affine_transform, transform_preds
from .visual import coco_box_to_bbox

try:
    from nms import soft_nms_39
except ImportError:
    print('NMS not installed! Do \n cd $CenterNet_ROOT/scripts/ \n'
          'and see run_standalone_eval.sh for more details to install it\n')

_NUM_JOINTS = config.num_joints


def post_process(dets, meta, scale=1):
    """rescale detection to original scale"""
    c, s, h, w = meta['c'], meta['s'], meta['out_height'], meta['out_width']
    b, K, N = dets.shape
    assert b == 1, "only single image was post-processed"
    dets = dets.reshape((K, N))
    bbox = transform_preds(dets[:, :4].reshape(-1, 2), c, s, (w, h)) / scale
    pts = transform_preds(dets[:, 5:39].reshape(-1, 2), c, s, (w, h)) / scale
    top_preds = np.concatenate(
        [bbox.reshape(-1, 4), dets[:, 4:5],
         pts.reshape(-1, 34)], axis=1).astype(np.float32).reshape(-1, 39)
    return top_preds


def merge_outputs(detections, soft_nms=True):
    """merge detections together by nms"""
    results = np.concatenate([detection for detection in detections], axis=0).astype(np.float32)
    if soft_nms:
        soft_nms_39(results, Nt=0.5, threshold=0.01, method=2)
    results = results.tolist()
    return results


def convert_eval_format(detections, img_id):
    """convert detection to annotation json format"""
    # detections. scores: (b, K); bboxes: (b, K, 4); kps: (b, K, J * 2); clses: (b, K)
    detections = np.array(detections).reshape((-1, 39))
    pred_anno = {"images": [], "annotations": []}
    num_objs, _ = detections.shape
    for i in range(num_objs):
        score = detections[i][4]
        bbox = detections[i][0:4]
        bbox[2:4] = bbox[2:4] - bbox[0:2]
        bbox = list(map(to_float, bbox))
        keypoints = np.concatenate([
            np.array(detections[i][5:39], dtype=np.float32).reshape(-1, 2),
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(_NUM_JOINTS * 3).tolist()
        keypoints = list(map(to_float, keypoints))
        class_id = 1
        pred = {
            "image_id": int(img_id),
            "category_id": int(class_id),
            "bbox": bbox,
            "score": to_float(score),
            "keypoints": keypoints
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
        pts = np.array(ann['keypoints'], np.float32).reshape(_NUM_JOINTS, 3)

        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        for j in range(_NUM_JOINTS):
            pts[j, :2] = affine_transform(pts[j, :2], trans_output)

        bbox = [bbox[0], bbox[1], w, h]
        keypoints = pts.reshape(_NUM_JOINTS * 3).tolist()
        ann["bbox"] = list(map(to_float, bbox))
        ann["keypoints"] = list(map(to_float, keypoints))
        resized_detection["annotations"].append(ann)
    return resize_detection
