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
"""export checkpoint file into air, onnx, mindir models"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.image_classification import CSPDarknet53

from model_utils.config import config
from model_utils.device_adapter import get_device_id

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    net = CSPDarknet53(num_classes=config.num_classes)
    param_dict = load_checkpoint(config.ckpt_file)
    param_dict_new = {}
    for k, v in param_dict.items():
        if k.startswith('moments.'):
            continue
        elif k.startswith('network.'):
            param_dict_new[k[8:]] = v
        else:
            param_dict_new[k] = v
    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    input_shape = [config.export_batch_size, 3, config.width, config.height]
    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=input_shape), ms.float32)

    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)
