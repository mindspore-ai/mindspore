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
'''
coco
'''
from __future__ import division

import json
import os
import pickle
from collections import defaultdict, OrderedDict
import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    has_coco = True
except ImportError:
    has_coco = False

from src.utils.nms import oks_nms

def _write_coco_keypoint_results(img_kpts, num_joints, res_file):
    '''
    _write_coco_keypoint_results
    '''
    results = []

    for img, items in img_kpts.items():
        item_size = len(items)
        if not items:
            continue
        kpts = np.array([items[k]['keypoints']
                         for k in range(item_size)])
        keypoints = np.zeros((item_size, num_joints * 3), dtype=np.float)
        keypoints[:, 0::3] = kpts[:, :, 0]
        keypoints[:, 1::3] = kpts[:, :, 1]
        keypoints[:, 2::3] = kpts[:, :, 2]

        result = [{'image_id': int(img),
                   'keypoints': list(keypoints[k]),
                   'score': items[k]['score'],
                   'category_id': 1,
                   } for k in range(item_size)]
        results.extend(result)

    with open(res_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


def _do_python_keypoint_eval(res_file, res_folder, ann_path):
    '''
    _do_python_keypoint_eval
    '''
    coco = COCO(ann_path)
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    eval_file = os.path.join(
        res_folder, 'keypoints_results.pkl')

    with open(eval_file, 'wb') as f:
        pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
    print('coco eval results saved to %s' % eval_file)

    return info_str

def evaluate(cfg, preds, output_dir, all_boxes, img_id, ann_path):
    '''
    evaluate
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    res_file = os.path.join(output_dir, 'keypoints_results.json')
    img_kpts_dict = defaultdict(list)
    for idx, file_id in enumerate(img_id):
        img_kpts_dict[file_id].append({
            'keypoints': preds[idx],
            'area': all_boxes[idx][0],
            'score': all_boxes[idx][1],
        })

    # rescoring and oks nms
    num_joints = cfg.MODEL.NUM_JOINTS
    in_vis_thre = cfg.TEST.IN_VIS_THRE
    oks_thre = cfg.TEST.OKS_THRE
    oks_nmsed_kpts = {}
    for img, items in img_kpts_dict.items():
        for item in items:
            kpt_score = 0
            valid_num = 0
            for n_jt in range(num_joints):
                max_jt = item['keypoints'][n_jt][2]
                if max_jt > in_vis_thre:
                    kpt_score = kpt_score + max_jt
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            item['score'] = kpt_score * item['score']
        keep = oks_nms(items, oks_thre)
        if not keep:
            oks_nmsed_kpts[img] = items
        else:
            oks_nmsed_kpts[img] = [items[kep] for kep in keep]

    # evaluate and save
    image_set = cfg.DATASET.TEST_SET
    _write_coco_keypoint_results(oks_nmsed_kpts, num_joints, res_file)
    if 'test' not in image_set and has_coco:
        ann_path = ann_path if ann_path else os.path.join(cfg.DATASET.ROOT, 'annotations',
                                                          'person_keypoints_' + image_set + '.json')
        info_str = _do_python_keypoint_eval(res_file, output_dir, ann_path)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']
    return {'Null': 0}, 0
