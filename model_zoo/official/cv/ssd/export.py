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

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.ssd import SSD300, SsdInferWithDecoder, ssd_mobilenet_v2, ssd_mobilenet_v1_fpn, ssd_resnet50_fpn, ssd_vgg16
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.box_utils import default_boxes

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num

    if config.model_name == "ssd300":
        net = SSD300(ssd_mobilenet_v2(), config, is_training=False)
    elif config.model_name == "ssd_vgg16":
        net = ssd_vgg16(config=config)
    elif config.model_name == "ssd_mobilenet_v1_fpn":
        net = ssd_mobilenet_v1_fpn(config=config)
    elif config.model_name == "ssd_resnet50_fpn":
        net = ssd_resnet50_fpn(config=config)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')

    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [config.batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mindspore.float32)
    export(net, input_array, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
