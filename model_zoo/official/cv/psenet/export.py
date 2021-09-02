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
##############export checkpoint file into air models#################
"""
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.model_utils.config import config
from src.PSENET.psenet import PSENet
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    net = PSENet(config)
    param_dict = load_checkpoint(config.ckpt)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([config.batch_size, 3, config.INFER_LONG_SIZE, config.INFER_LONG_SIZE]), ms.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
