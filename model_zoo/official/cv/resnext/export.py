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
"""
resnext export mindir.
"""
import os
import numpy as np
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.image_classification import get_network
from src.utils.auto_mixed_precision import auto_mixed_precision


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    network = get_network(network=config.network, num_classes=config.num_classes, platform=config.device_target)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(network, param_dict)
    if config.device_target == "Ascend":
        network.to_float(mstype.float16)
    else:
        auto_mixed_precision(network)
    network.set_train(False)
    input_shp = [config.batch_size, 3, config.height, config.width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(network, input_array, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
