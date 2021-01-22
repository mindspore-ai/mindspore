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
"""export checkpoint file into air models"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.config import config

parser = argparse.ArgumentParser(description='fasterrcnn_export')
parser.add_argument('--ckpt_file', type=str, default='', help='fasterrcnn ckpt file.')
parser.add_argument('--output_file', type=str, default='', help='fasterrcnn output air name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX"], default='AIR', help='file format')
args = parser.parse_args()

if __name__ == '__main__':
    net = Faster_Rcnn_Resnet50(config=config)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), ms.float16)
    img_metas = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, 4]), ms.float16)
    gt_bboxes = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, config.num_gts]), ms.float16)
    gt_label = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, config.num_gts]), ms.int32)
    gt_num = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, config.num_gts]), ms.bool_)

    export(net, img, img_metas, gt_bboxes, gt_label, gt_num, file_name=args.output_file, file_format=args.file_format)
