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
"""export checkpoint file into air models"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.warpctc import StackedRNN
from src.config import config

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='warpctc_export')
    parser.add_argument('--ckpt_file', type=str, default='', help='warpctc ckpt file.')
    parser.add_argument('--output_file', type=str, default='', help='warpctc output air name.')
    args_opt = parser.parse_args()
    captcha_width = config.captcha_width
    captcha_height = config.captcha_height
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    net = StackedRNN(captcha_height * 3, batch_size, hidden_size)
    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    image = Tensor(np.zeros([batch_size, 3, captcha_height, captcha_width], np.float16))
    export(net, image, file_name=args_opt.output_file, file_format="AIR")
