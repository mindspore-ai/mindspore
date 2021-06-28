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
"""eval Xception."""
import os
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.Xception import xception
from src.model_utils.config import config as args, config_gpu, config_ascend
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_file)
    args.file_name = os.path.join(args.output_path, args.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export function'''
    if args.device_target == "Ascend":
        config = config_ascend
    elif args.device_target == "GPU":
        config = config_gpu
    else:
        raise ValueError("Unsupported device_target.")

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(device_id=args.device_id)
    net = xception(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    image = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, image, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    run_export()
