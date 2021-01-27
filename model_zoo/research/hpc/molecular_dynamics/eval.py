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
"""eval."""
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.network import Network

parser = argparse.ArgumentParser(description='MD Simulation')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend")

if __name__ == '__main__':
    # get input data
    r = np.load(args_opt.dataset_path)
    d_coord, d_nlist, avg, std, atype, nlist = r['d_coord'], r['d_nlist'], r['avg'], r['std'], r['atype'], r['nlist']
    batch_size = 1
    atype_tensor = Tensor(atype)
    avg_tensor = Tensor(avg)
    std_tensor = Tensor(std)
    nlist_tensor = Tensor(nlist)
    d_coord_tensor = Tensor(np.reshape(d_coord, (1, -1, 3)))
    d_nlist_tensor = Tensor(d_nlist)
    frames = []
    for i in range(batch_size):
        frames.append(i * 1536)
    frames = Tensor(frames)
    # evaluation
    net = Network()
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.to_float(mstype.float32)
    energy, atom_ener, _ = \
        net(d_coord_tensor, d_nlist_tensor, frames, avg_tensor, std_tensor, atype_tensor, nlist_tensor)
    print('energy:', energy)
    print('atom_energy:', atom_ener)
