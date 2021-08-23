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
""" export script """


import numpy as np

import mindspore
from mindspore import context, Tensor, export
from mindspore.train.serialization import load_checkpoint
from src.net import SiameseRPN


def siamrpn_export():
    """ export function """
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        save_graphs=False,
        device_id=args.device_id)
    net = SiameseRPN(groups=1)
    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)
    input_data1 = Tensor(np.zeros([1, 3, 127, 127]), mindspore.float32)
    input_data2 = Tensor(np.zeros([1, 3, 255, 255]), mindspore.float32)
    input_data = [input_data1, input_data2]
    export(net, *input_data, file_name='siamrpn3', file_format="MINDIR")


if __name__ == '__main__':
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument('--ckpt_file', type=str, required=True, help='siamRPN ckpt file.')
    args = parser.parse_args()
    siamrpn_export()
