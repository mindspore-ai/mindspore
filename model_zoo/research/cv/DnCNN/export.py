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

"""export file of MINDIR format"""

import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.model import DnCNN

parse = argparse.ArgumentParser(description='DnCNN export')
parse.add_argument("--batch_size", type=int, default=128, help="batch size")
parse.add_argument("--image_height", type=int, default=256, help="height of each input image")
parse.add_argument("--image_width", type=int, default=256, help="width of each input image")
parse.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint file path.")
parse.add_argument("--file_name", type=str, default="DnCNN", help="output file name.")
parse.add_argument("--file_format", type=str, default="MINDIR", help="output file format")
args = parse.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dncnn = DnCNN()
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(dncnn, param_dict)
    input_arr = Tensor(np.ones([args.batch_size, 1, args.image_height, args.image_width]), ms.float32)
    export(dncnn, input_arr, file_name=args.file_name, file_format=args.file_format)
