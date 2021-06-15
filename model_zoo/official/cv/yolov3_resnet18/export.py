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

import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.yolov3 import yolov3_resnet18, YoloWithEval
from src.config import ConfigYOLOV3ResNet18

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    default_config.file_name = os.path.join(default_config.output_path, default_config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    context.set_context(mode=context.GRAPH_MODE, device_target=default_config.device_target)
    if default_config.device_target == "Ascend":
        context.set_context(device_id=default_config.device_id)
    cfg = ConfigYOLOV3ResNet18()
    net = yolov3_resnet18(cfg)
    eval_net = YoloWithEval(net, cfg)

    param_dict = load_checkpoint(default_config.ckpt_file)
    load_param_into_net(eval_net, param_dict)

    eval_net.set_train(False)

    shape = [default_config.export_batch_size, 3] + cfg.img_shape
    input_data = Tensor(np.zeros(shape), ms.float32)
    input_shape = Tensor(np.zeros([1, 2]), ms.float32)
    inputs = (input_data, input_shape)

    export(eval_net, *inputs, file_name=default_config.file_name, file_format=default_config.file_format)


if __name__ == "__main__":
    run_export()
