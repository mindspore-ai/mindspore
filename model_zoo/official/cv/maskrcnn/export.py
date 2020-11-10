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

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.maskrcnn.mask_rcnn_r50 import Mask_Rcnn_Resnet50
from src.config import config

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='maskrcnn_export')
    parser.add_argument('--ckpt_file', type=str, default='', help='maskrcnn ckpt file.')
    parser.add_argument('--output_file', type=str, default='', help='maskrcnn output air name.')
    args_opt = parser.parse_args()

    net = Mask_Rcnn_Resnet50(config=config)
    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    bs = config.test_batch_size

    img = Tensor(np.zeros([bs, 3, 768, 1280], np.float16))
    img_metas = Tensor(np.zeros([bs, 4], np.float16))
    gt_bboxes = Tensor(np.zeros([bs, 128, 4], np.float16))
    gt_labels = Tensor(np.zeros([bs, 128], np.int32))
    gt_num = Tensor(np.zeros([bs, 128], np.bool))
    gt_mask = Tensor(np.zeros([bs, 128], np.bool))
    export(net, img, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask, file_name=args_opt.output_file,
           file_format="AIR")
