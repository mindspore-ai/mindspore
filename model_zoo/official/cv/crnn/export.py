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

""" export model for CRNN """
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, export
from src.crnn import crnn
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    config.batch_size = 1
    net = crnn(config)

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, config.image_height, config.image_width]), ms.float32)

    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
