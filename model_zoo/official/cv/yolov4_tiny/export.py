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
import argparse
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.yolo import YOLOV4TinyCspDarkNet53

from model_utils.config import config

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='checkpoint export')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint of yolov4_tiny (Default: None)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_classes', type=int, default=80, help='the number of classes (Default: 81)')
    parser.add_argument('--file_name', type=str, default='yolov4_tiny', help='output file name')
    args = parser.parse_args()

    ts_shape = config.testing_shape
    config.ckpt_file = args.checkpoint
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes

    network = YOLOV4TinyCspDarkNet53()
    network.set_train(False)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)

    input_data = Tensor(np.zeros([config.batch_size, 3, ts_shape, ts_shape]), mindspore.float32)

    export(network, input_data, file_name=args.file_name, file_format=config.file_format)
    print('export sucess')
