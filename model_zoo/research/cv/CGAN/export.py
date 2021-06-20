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
"""export"""
import argparse
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.model import Generator


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt', help='checkpoint dir of CGAN')
    args = parser.parse_args()
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE, device_target="Ascend")
    return args

def main():
    # before training, we should set some arguments
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    args = preLauch()

    # training argument
    input_dim = 100
    # create G Cell & D Cell
    netG = Generator(input_dim)

    latent_code_eval = Tensor(np.random.randn(200, input_dim), dtype=mstype.float32)

    label_eval = np.zeros((200, 10))
    for i in range(200):
        j = i // 20
        label_eval[i][j] = 1
    label_eval = Tensor(label_eval, dtype=mstype.float32)

    param_G = load_checkpoint(args.ckpt_dir)
    load_param_into_net(netG, param_G)
    netG.set_train(False)
    export(netG, latent_code_eval, label_eval, file_name="CGAN", file_format="MINDIR")
    print("CGAN exported")

if __name__ == '__main__':
    main()
