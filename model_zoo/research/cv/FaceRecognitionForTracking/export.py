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
"""Convert ckpt to air/mindir."""
import os
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.reid import SphereNet_float32

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    if config.device_target == 'Ascend':
        devid = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=devid)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    network = SphereNet_float32(num_layers=12, feature_dim=128, shape=(96, 64))
    ckpt_path = config.pretrained
    if os.path.isfile(ckpt_path):
        param_dict = load_checkpoint(ckpt_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('model.'):
                param_dict_new[key[6:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('-----------------------load model success-----------------------')
    else:
        print('-----------------------load model failed -----------------------')

    if config.device_target == 'CPU':
        network.add_flags_recursive(fp32=True)
    else:
        network.add_flags_recursive(fp16=True)
    network.set_train(False)

    input_data = np.random.uniform(low=0, high=1.0, size=(config.batch_size, 3, 96, 64)).astype(np.float32)
    tensor_input_data = Tensor(input_data)

    export(network, tensor_input_data, file_name=config.file_name, file_format=config.file_format)
    print('-----------------------export model success-----------------------')


if __name__ == "__main__":
    run_export()
