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
import numpy as np

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.centerface import CenterfaceMobilev2, CenterFaceWithNms
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_centerface():
    net = CenterfaceMobilev2()

    param_dict = load_checkpoint(config.ckpt_file)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
            continue
        elif key.startswith('centerface_network.'):
            param_dict_new[key[19:]] = values
        else:
            param_dict_new[key] = values

    load_param_into_net(net, param_dict_new)
    net = CenterFaceWithNms(net)
    net.set_train(False)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.input_h, config.input_w]), mindspore.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_centerface()
