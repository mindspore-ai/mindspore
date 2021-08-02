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
from PIL import Image
from eval import DetectionEngine

def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size

parser = argparse.ArgumentParser('YoloV3 postprocess')
parser.add_argument('--result_path', type=str, required=True, help='result files path.')
parser.add_argument('--img_path', type=str, required=True, help='train data dir.')
parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')
parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
parser.add_argument('--ann_file', type=str, default='', help='path to annotation')
parser.add_argument('--eval_ignore_threshold', type=float, default=0.001, help='threshold to throw low quality boxes')
parser.add_argument('--log_path', type=str, default='outputs/', help='inference result save location')
parser.add_argument('--multi_label', type=bool, default=True, help='whether to use multi label')
parser.add_argument('--multi_label_thresh', type=float, default=0.15, help='threshhold to throw low quality boxes')
args = parser.parse_args()

if __name__ == "__main__":
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    detection = DetectionEngine(args)
    bs = args.per_batch_size

    f_list = os.listdir(args.img_path)
    for f in f_list:
        image_size = get_img_size(os.path.join(args.img_path, f))
        f = f.split('.')[0]
        output_big = np.fromfile(os.path.join(args.result_path, f + '_0.bin'), np.float32).reshape(bs, 20, 20, 3, 85)
        output_small = np.fromfile(os.path.join(args.result_path, f + '_1.bin'), np.float32).reshape(bs, 40, 40, 3, 85)
        image_id = [int(f.split('_')[-1])]
        image_shape = [[image_size[0], image_size[1]]]
        detection.detect([output_small, output_big], bs, image_shape, image_id)

    print('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    print('result file path: {}'.format(result_file_path))
    eval_result = detection.get_eval_result()

    print('\n=============coco eval result=========\n' + eval_result)
