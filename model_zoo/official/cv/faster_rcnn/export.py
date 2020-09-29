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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fasterrcnn_export')
    parser.add_argument('--ckpt_file', type=str, default='', help='fasterrcnn ckpt file.')
    parser.add_argument('--output_file', type=str, default='', help='fasterrcnn output air name.')
    args_opt = parser.parse_args()

    net = Faster_Rcnn_Resnet50(config=config)

    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(net, param_dict)

    img = Tensor(np.random.uniform(0.0, 1.0, size=[1, 3, 768, 1280]), ms.float16)
    img_shape = Tensor(np.random.uniform(0.0, 1.0, size=[768, 1280, 1]), ms.float16)
    gt_bboxes = Tensor(np.random.uniform(0.0, 1.0, size=[1, 128]), ms.float16)
    gt_label = Tensor(np.random.uniform(0.0, 1.0, size=[1, 128]), ms.int32)
    gt_num = Tensor(np.random.uniform(0.0, 1.0, size=[1, 128]), ms.bool)
    export(net, img, img_shape, gt_bboxes, gt_label, gt_num, file_name=args_opt.output_file, file_format="AIR")
