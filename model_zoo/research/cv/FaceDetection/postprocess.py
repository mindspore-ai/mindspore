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
import matplotlib.pyplot as plt
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from src.FaceDetection import voc_wrapper
from src.network_define import get_bounding_boxes, tensor_to_brambox, \
    parse_gt_from_anno, parse_rets, calc_recall_precision_ap

from model_utils.config import config


def cal_map(result_path, data_dir, save_output_path):
    """cal map"""
    labels = ['face']
    det = {}
    img_size = {}
    img_anno = {}
    eval_times = 0
    classes = {0: 'face'}
    ret_files_set = {'face': os.path.join(save_output_path, 'comp4_det_test_face_rm5050.txt')}
    files = os.listdir(os.path.join(data_dir, "labels"))
    for file in files:
        image_name = file.split('.')[0]
        label = np.fromfile(os.path.join(data_dir, "labels", file), dtype=np.float64).reshape((1, 200, 6))
        image_size = np.fromfile(os.path.join(data_dir, "image_size", file), dtype=np.int32).reshape((1, 1, 2))
        eval_times += 1
        dets = []
        tdets = []
        file_path = os.path.join(result_path, image_name)
        coords_0 = np.fromfile(file_path + '_0.bin', dtype=np.float32).reshape((1, 4, 84, 4))
        coords_0 = Tensor(coords_0, mstype.float32)
        cls_scores_0 = np.fromfile(file_path + '_1.bin', dtype=np.float32).reshape((1, 4, 84))
        cls_scores_0 = Tensor(cls_scores_0, mstype.float32)
        coords_1 = np.fromfile(file_path + '_2.bin', dtype=np.float32).reshape((1, 4, 336, 4))
        coords_1 = Tensor(coords_1, mstype.float32)
        cls_scores_1 = np.fromfile(file_path + '_3.bin', dtype=np.float32).reshape((1, 4, 336))
        cls_scores_1 = Tensor(cls_scores_1, mstype.float32)
        coords_2 = np.fromfile(file_path + '_4.bin', dtype=np.float32).reshape((1, 4, 1344, 4))
        coords_2 = Tensor(coords_2, mstype.float32)
        cls_scores_2 = np.fromfile(file_path + '_5.bin', dtype=np.float32).reshape((1, 4, 1344))
        cls_scores_2 = Tensor(cls_scores_2, mstype.float32)

        boxes_0, boxes_1, boxes_2 = get_bounding_boxes(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2,
                                                       cls_scores_2, config.conf_thresh, config.input_shape,
                                                       config.num_classes)

        converted_boxes_0, converted_boxes_1, converted_boxes_2 = tensor_to_brambox(boxes_0, boxes_1, boxes_2,
                                                                                    config.input_shape, labels)

        tdets.append(converted_boxes_0)
        tdets.append(converted_boxes_1)
        tdets.append(converted_boxes_2)
        batch = len(tdets[0])
        for b in range(batch):
            single_dets = []
            for op in range(3):
                single_dets.extend(tdets[op][b])
            dets.append(single_dets)

        det.update({image_name: v for k, v in enumerate(dets)})
        img_size.update({image_name: v for k, v in enumerate(image_size)})
        img_anno.update({image_name: v for k, v in enumerate(label)})

    netw, neth = config.input_shape
    reorg_dets = voc_wrapper.reorg_detection(det, netw, neth, img_size)
    voc_wrapper.gen_results(reorg_dets, save_output_path, img_size, config.nms_thresh)

    # compute mAP
    ground_truth = parse_gt_from_anno(img_anno, classes)

    ret_list = parse_rets(ret_files_set)
    iou_thr = 0.5
    evaluate = calc_recall_precision_ap(ground_truth, ret_list, iou_thr)
    print(evaluate)

    aps_str = ''
    for cls in evaluate:
        per_line, = plt.plot(evaluate[cls]['recall'], evaluate[cls]['precision'], 'b-')
        per_line.set_label('%s:AP=%.4f' % (cls, evaluate[cls]['ap']))
        aps_str += '_%s_AP_%.4f' % (cls, evaluate[cls]['ap'])
        plt.plot([i / 1000.0 for i in range(1, 1001)], [i / 1000.0 for i in range(1, 1001)], 'y--')
        plt.axis([0, 1.2, 0, 1.2])  # [x_min, x_max, y_min, y_max]
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()

        plt.legend()
        plt.title('PR')

    # save mAP
    ap_save_path = os.path.join(save_output_path, save_output_path.replace('/', '_') + aps_str + '.png')
    print('Saving {}'.format(ap_save_path))
    plt.savefig(ap_save_path)

    print('=============yolov3 evaluating finished==================')


if __name__ == '__main__':
    cal_map(config.result_path, config.data_dir, config.save_output_path)
