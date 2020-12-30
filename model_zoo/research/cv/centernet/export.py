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
Export CenterNet mindir model.
"""

import argparse
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src import CenterNetMultiPoseEval
from src.config import net_config, eval_config, export_config

parser = argparse.ArgumentParser(description='centernet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

if __name__ == '__main__':
    net = CenterNetMultiPoseEval(net_config, eval_config.K)
    net.set_train(False)

    param_dict = load_checkpoint(export_config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shape = [1, 3, export_config.input_res[0], export_config.input_res[1]]
    input_data = Tensor(np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32))

    export(net, input_data, file_name=export_config.export_name, file_format=export_config.export_format)
