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
"""post process for 310 inference"""
import os
import argparse
import numpy as np
from pycocotools.coco import COCO
from util import coco_eval, bbox2result_1image, results2json, get_seg_masks


parser = argparse.ArgumentParser("maskrcnn quant postprocess")
parser.add_argument("--shape_path", type=str, required=True, help="path to image meta directory")
parser.add_argument("--annotation_path", type=str, required=True, help="path to instance_xxx.json")
parser.add_argument("--result_path", type=str, required=True, help="path to inference results.")

args, _ = parser.parse_known_args()


def get_eval_result(shape_data, ann_file, result_path):
    """ Get metrics result according to the annotation file and result file"""
    max_num = 128
    result_path = result_path
    outputs = []

    dataset_coco = COCO(ann_file)
    for index in range(len(os.listdir(shape_data))):
        prefix = "coco2017_maskrcnn_bs_1_"
        shape_file_path = os.path.join(shape_data, prefix + str(index) + ".bin")
        shape_file = np.fromfile(shape_file_path, dtype=np.float16).reshape(1, 4)

        bbox_result_file = os.path.join(result_path, prefix + str(index) + "_output_0.bin")
        label_result_file = os.path.join(result_path, prefix + str(index) + "_output_1.bin")
        mask_result_file = os.path.join(result_path, prefix + str(index) + "_output_2.bin")
        mask_fb_result_file = os.path.join(result_path, prefix + str(index) + "_output_3.bin")

        all_bbox = np.fromfile(bbox_result_file, dtype=np.float16).reshape(80000, 5)
        all_label = np.fromfile(label_result_file, dtype=np.int32).reshape(80000, 1)
        all_mask = np.fromfile(mask_result_file, dtype=np.bool_).reshape(80000, 1)
        all_mask_fb = np.fromfile(mask_fb_result_file, dtype=np.float16).reshape(80000, 28, 28)

        all_bbox_squee = np.squeeze(all_bbox)
        all_label_squee = np.squeeze(all_label)
        all_mask_squee = np.squeeze(all_mask)
        all_mask_fb_squee = np.squeeze(all_mask_fb)

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]
        all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :]

        if all_bboxes_tmp_mask.shape[0] > max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]
            all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]
        num_classes = 81
        bbox_results = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, num_classes)
        segm_results = get_seg_masks(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask, shape_file[0],
                                     True, num_classes)
        outputs.append((bbox_results, segm_results))

    eval_types = ["bbox", "segm"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    coco_eval(result_files, eval_types, dataset_coco, single_result=False)


if __name__ == '__main__':
    get_eval_result(args.shape_path, args.annotation_path, args.result_path)
