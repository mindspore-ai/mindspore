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
"""YoloV3 postprocess."""
import os
import argparse
import datetime
import numpy as np
from eval import DetectionEngine

parser = argparse.ArgumentParser('YoloV3_quant postprocess')
parser.add_argument('--result_path', type=str, required=True, help='result files path.')
parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')
parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
parser.add_argument('--annFile', type=str, default='', help='path to annotation')
parser.add_argument('--image_shape', type=str, default='./image_shape.npy', help='path to image_shape.npy')
parser.add_argument('--image_id', type=str, default='./image_id.npy', help='path to image_id.npy')
parser.add_argument('--ignore_threshold', type=float, default=0.001, help='threshold to throw low quality boxes')
parser.add_argument('--log_path', type=str, default='outputs/', help='inference result save location')

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    detection = DetectionEngine(args)
    bs = args.per_batch_size

    f_list = os.listdir(args.result_path)
    shape_list = np.load(args.image_shape)
    id_list = np.load(args.image_id)
    prefix = "YoloV3-DarkNet_coco_bs_" + str(bs) + "_"
    iter_num = 0
    for image_id in id_list:
        image_shape = shape_list[iter_num]
        path_small = os.path.join(args.result_path, prefix + str(iter_num) + '_0.bin')
        path_medium = os.path.join(args.result_path, prefix + str(iter_num) + '_1.bin')
        path_big = os.path.join(args.result_path, prefix + str(iter_num) + '_2.bin')
        if os.path.exists(path_small) and os.path.exists(path_medium) and os.path.exists(path_big):
            output_small = np.fromfile(path_small, np.float32).reshape(bs, 13, 13, 3, 85)
            output_medium = np.fromfile(path_medium, np.float32).reshape(bs, 26, 26, 3, 85)
            output_big = np.fromfile(path_big, np.float32).reshape(bs, 52, 52, 3, 85)
            detection.detect([output_small, output_medium, output_big], bs, image_shape, image_id)
        else:
            print("Error: Image ", iter_num, " is not exist.")
        iter_num += 1

    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    eval_result = detection.get_eval_result()

    print('\n=============coco eval result=========\n' + eval_result)
