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
"""export checkpoint file into air models"""
import math as m
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.warpctc import StackedRNN, StackedRNNForGPU, StackedRNNForCPU
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

if config.file_format == "AIR" and config.device_target != "Ascend":
    raise ValueError("export AIR must on Ascend")

if __name__ == "__main__":
    input_size = m.ceil(config.captcha_height / 64) * 64 * 3
    captcha_width = config.captcha_width
    captcha_height = config.captcha_height
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    image = Tensor(np.zeros([batch_size, 3, captcha_height, captcha_width], np.float32))
    if config.device_target == 'Ascend':
        net = StackedRNN(input_size=input_size, batch_size=batch_size, hidden_size=hidden_size)
        image = Tensor(np.zeros([batch_size, 3, captcha_height, captcha_width], np.float16))
    elif config.device_target == 'GPU':
        net = StackedRNNForGPU(input_size=input_size, batch_size=batch_size, hidden_size=hidden_size)
    else:
        net = StackedRNNForCPU(input_size=input_size, batch_size=batch_size, hidden_size=hidden_size)
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    export(net, image, file_name=config.file_name, file_format=config.file_format)
