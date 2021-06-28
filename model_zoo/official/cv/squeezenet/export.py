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
"""
##############export checkpoint file into air , mindir and onnx models#################
python export.py --net squeezenet --dataset cifar10 --checkpoint_path squeezenet_cifar10-120_1562.ckpt
"""
import os
import numpy as np
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

if config.net_name == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
if config.dataset == "cifar10":
    num_classes = 10
else:
    num_classes = 1000

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    net = squeezenet(num_classes=num_classes)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.height, config.width], np.float32))
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
