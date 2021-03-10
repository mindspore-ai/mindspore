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
"""export for retinanet"""
import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.retinanet import  retinanet50, resnet50, retinanetInferWithDecoder
from src.config import config
from src.box_utils import default_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retinanet evaluation')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend"),
                        help="run platform, only support Ascend.")
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--file_name", type=str, default="retinanet", help="output file name.")
    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=args_opt.device_id)

    backbone = resnet50(config.num_classes)
    net = retinanet50(backbone, config)
    net = retinanetInferWithDecoder(net, Tensor(default_boxes), config)
    param_dict = load_checkpoint(config.checkpoint_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)
    shape = [args_opt.batch_size, 3] + config.img_shape
    input_data = Tensor(np.zeros(shape), mstype.float32)
    export(net, input_data, file_name=args_opt.file_name, file_format=args_opt.file_format)
