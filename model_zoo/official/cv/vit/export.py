# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
##############export checkpoint file into air and onnx models#################
python export.py
"""
import os
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.vit import get_network

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    net = get_network(backbone_name=config.backbone, args=config)

    assert config.pretrained is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(config.pretrained)
    load_param_into_net(net, param_dict)

    config.height = config.train_image_size
    config.width = config.train_image_size

    input_arr = Tensor(np.zeros([config.batch_size, 3, config.height, config.width], np.float32))
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
