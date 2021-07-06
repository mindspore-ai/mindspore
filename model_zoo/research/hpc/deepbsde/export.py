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
"""DeepBSDE export model script"""
import os
from mindspore import context, load_checkpoint, export, Tensor
from src.net import DeepBSDE
from src.config import config
from src.equation import get_bsde

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.ckpt_path = os.path.join(config.log_dir, "deepbsde_{}_end.ckpt".format(config.eqn_name))
    bsde = get_bsde(config)
    print('Begin to solve', config.eqn_name)
    net = DeepBSDE(config, bsde)
    load_checkpoint(config.ckpt_path, net=net)
    dw, x = bsde.sample(config.valid_size)
    export(net, Tensor(dw), Tensor(x), file_name="deepbsde_{}".format(config.eqn_name), file_format=config.file_format)
