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
"""export script"""
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, export
from src.cnn_direction_model import CNNDirectionModel
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


device_id = get_device_id()
context.set_context(device_id=device_id)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    net = CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512])

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, config.im_size_h, config.im_size_w]), ms.float32)

    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
