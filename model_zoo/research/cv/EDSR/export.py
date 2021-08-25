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
"""
##############export checkpoint file into air, mindir models#################
python export.py
"""
import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, export, context

from src.utils import init_net
from model_utils.config import config
from model_utils.device_adapter import get_device_id
from model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

MAX_HR_SIZE = 2040


@moxing_wrapper()
def run_export():
    """
    run export
    """
    print(config)

    cfg = config
    if cfg.pre_trained is None:
        raise RuntimeError('config.pre_trained is None.')

    net = init_net(cfg)
    max_lr_size = MAX_HR_SIZE // cfg.scale
    input_arr = Tensor(np.ones([1, cfg.n_colors, max_lr_size, max_lr_size]), ms.float32)
    file_name = os.path.splitext(os.path.basename(cfg.pre_trained))[0]
    file_name = file_name + f"_InputSize{max_lr_size}"
    file_path = os.path.join(cfg.output_path, file_name)
    file_format = 'MINDIR'

    num_params = sum([param.size for param in net.parameters_dict().values()])
    export(net, input_arr, file_name=file_path, file_format=file_format)
    print(f"export success", flush=True)
    print(f"{cfg.pre_trained} -> {file_path}.{file_format.lower()}, net parameters = {num_params/1000000:>0.4}M",
          flush=True)

if __name__ == '__main__':
    run_export()
