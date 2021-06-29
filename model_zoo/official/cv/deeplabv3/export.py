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
"""export checkpoint file into air, onnx, mindir models"""
import os
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.nets import net_factory

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper

class BuildEvalNetwork(nn.Cell):
    def __init__(self, net, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = net
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, x):
        if self.format == "NHWC":
            x = self.transpose(x, (0, 3, 1, 2))
        output = self.network(x)
        output = self.softmax(output)
        return output


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    if config.export_model == 'deeplab_v3_s16':
        network = net_factory.nets_map['deeplab_v3_s16']('eval', config.num_classes, 16, True)
    else:
        network = net_factory.nets_map['deeplab_v3_s8']('eval', config.num_classes, 8, True)
    network = BuildEvalNetwork(network, config.input_format)
    param_dict = load_checkpoint(config.ckpt_file)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    if config.input_format == "NHWC":
        input_data = Tensor(
            np.ones([config.export_batch_size, config.input_size, config.input_size, 3]).astype(np.float32))
    else:
        input_data = Tensor(
            np.ones([config.export_batch_size, 3, config.input_size, config.input_size]).astype(np.float32))
    export(network, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
