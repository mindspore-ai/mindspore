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
"""Convert ckpt to air."""
import os
import argparse
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.centerface import CenterfaceMobilev2

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

def save_air():
    """Save air file"""
    print('============= centerface start save air ==================')

    parser = argparse.ArgumentParser(description='Convert ckpt to air')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args = parser.parse_args()
    network = CenterfaceMobilev2()

    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
                continue
            elif key.startswith('centerface_network.'):
                param_dict_new[key[19:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('load model {} success'.format(args.pretrained))

        input_data = np.random.uniform(low=0, high=1.0, size=(args.batch_size, 3, 832, 832)).astype(np.float32)

        tensor_input_data = Tensor(input_data)
        export(network, tensor_input_data,
               file_name=args.pretrained.replace('.ckpt', '_' + str(args.batch_size) + 'b.air'), file_format='AIR')

        print("export model success.")


if __name__ == "__main__":
    save_air()
