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
"""export checkpoint file into air, mindir models"""
import os
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.Deeptext.deeptext_vgg16 import Deeptext_VGG16_Infer

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id



def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    config.test_batch_size = config.export_batch_size
    context.set_context(mode=context.GRAPH_MODE, device_target=config.export_device_target)
    context.set_context(device_id=get_device_id())

    net = Deeptext_VGG16_Infer(config=config)
    net.set_train(False)

    param_dict = load_checkpoint(config.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    img_data = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), ms.float32)

    export(net, img_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
