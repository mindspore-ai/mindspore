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

"""export checkpoint file into air, onnx, mindir models
   Suggest run as python export.py --file_name [file_name] --ckpt_files [ckpt path] --file_format [file format]
"""
import os
import numpy as np
from mindspore.common import dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    if config.net == "densenet100":
        from src.network.densenet import DenseNet100 as DenseNet
    else:
        from src.network.densenet import DenseNet121 as DenseNet

    network = DenseNet(config.num_classes)

    param_dict = load_checkpoint(config.ckpt_files)

    param_dict_new = {}
    for key, value in param_dict.items():
        if key.startswith("moments."):
            continue
        elif key.startswith("network."):
            param_dict_new[key[8:]] = value
        else:
            param_dict_new[key] = value

    load_param_into_net(network, param_dict_new)

    network.add_flags_recursive(fp16=True)
    network.set_train(False)

    shape = [int(config.batch_size), 3] + [int(config.image_size.split(",")[0]), int(config.image_size.split(",")[1])]
    input_data = Tensor(np.zeros(shape), mstype.float32)

    export(network, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
