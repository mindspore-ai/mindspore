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

import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, export

from src.maskrcnn_mobilenetv1.mask_rcnn_mobilenetv1 import Mask_Rcnn_Mobilenetv1
from src.config import config

parser = argparse.ArgumentParser(description="maskrcnn mobilnetv1 export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="maskrcnn_mobilenetv1", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config.test_batch_size = args.batch_size
    net = Mask_Rcnn_Mobilenetv1(config)
    load_checkpoint(args.ckpt_file, net=net)

    net.set_train(False)

    img_data = Tensor(np.zeros([args.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([args.batch_size, 4], np.float16))
    gt_bboxes = Tensor(np.zeros([args.batch_size, config.num_gts, 4], np.float16))
    gt_labels = Tensor(np.zeros([args.batch_size, config.num_gts], np.int32))
    gt_num = Tensor(np.zeros([args.batch_size, config.num_gts], np.bool))
    gt_mask = Tensor(np.zeros([args.batch_size, 1, 1, 1], np.bool))

    input_data = [img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask]
    export(net, *input_data, file_name=args.file_name, file_format=args.file_format)
