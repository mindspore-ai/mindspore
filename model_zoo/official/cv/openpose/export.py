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
"""export"""

import argparse
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.openposenet import OpenPoseNet

parser = argparse.ArgumentParser(description='checkpoint export')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
args_opt = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    # define net
    net = OpenPoseNet()

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    inputs = np.random.uniform(0.0, 1.0, size=[1, 3, 368, 368]).astype(np.float32)
    export(net, Tensor(inputs), file_name="openpose.air", file_format='AIR')
