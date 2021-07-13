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
export model
"""
import os

import numpy as np
from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net

from src.config import parse_args
from src.models.StackedHourglassNet import StackedHourglassNet

args = parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.ckpt_file):
        print("ckpt file not valid")
        exit()

    # Set context mode
    if args.context_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    # Import net
    net = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args.batch_size, args.input_res, args.input_res, 3], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
