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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import json
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.args import get_args
from src.dcgan_model import DcganG
from src.dcgannobn_model import DcgannobnG

if __name__ == '__main__':
    args_opt = get_args('export')
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)

    with open(args_opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())

    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    n_extra_layers = generator_config["n_extra_layers"]

    # generator
    if noBN:
        netG = DcgannobnG(imageSize, nz, nc, ngf, n_extra_layers)
    else:
        netG = DcganG(imageSize, nz, nc, ngf, n_extra_layers)

    # load weights
    load_param_into_net(netG, load_checkpoint(args_opt.ckpt_file))

    # initialize noise
    fixed_noise = Tensor(np.random.normal(size=[args_opt.nimages, nz, 1, 1]), dtype=mstype.float32)

    export(netG, fixed_noise, file_name=args_opt.file_name, file_format=args_opt.file_format)
