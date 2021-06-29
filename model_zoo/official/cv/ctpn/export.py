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
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.ctpn import CTPN_Infer
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    config.feature_shapes = [config.img_height // 16, config.img_width // 16]
    config.num_bboxes = (config.img_height // 16) * (config.img_width // 16) * config.num_anchors
    config.num_step = config.img_width // 16
    config.rnn_batch_size = config.img_height // 16

    net = CTPN_Infer(config=config, batch_size=config.test_batch_size)

    param_dict = load_checkpoint(config.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), ms.float16)

    export(net, img, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
