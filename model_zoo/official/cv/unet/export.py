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

import os
import numpy as np

from mindspore import Tensor, export, load_checkpoint, load_param_into_net, context

from src.unet_medical.unet_model import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.utils import UnetEval
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)
    elif config.model_name == 'unet_nested':
        net = NestedUNet(in_channel=config.num_channels, n_class=config.num_classes, use_deconv=config.use_deconv,
                         use_bn=config.use_bn, use_ds=False)
    elif config.model_name == 'unet_simple':
        net = UNet(in_channel=config.num_channels, n_class=config.num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(config.model_name))
    # return a parameter dict for model
    param_dict = load_checkpoint(config.checkpoint_file_path)
    # load the parameter into net
    load_param_into_net(net, param_dict)
    net = UnetEval(net, eval_activate=config.eval_activate.lower())
    input_data = Tensor(np.ones([config.batch_size, config.num_channels, config.height, \
        config.width]).astype(np.float32))
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
