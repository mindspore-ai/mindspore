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
##############export checkpoint file into air and onnx models#################
python export.py
"""
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.model_utils.config import config
from src.tinydarknet import TinyDarkNet
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)
    config.checkpoint_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """export function"""
    if config.dataset_name != 'imagenet':
        raise ValueError("Dataset is not support.")

    net = TinyDarkNet(num_classes=config.num_classes)

    assert config.checkpoint_path is not None, "config.checkpoint_path is None."
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[config.batch_size, 3, 224, 224]), ms.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
