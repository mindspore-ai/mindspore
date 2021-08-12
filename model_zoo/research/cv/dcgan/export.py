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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import ast
import os

import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
import mindspore.common.dtype as mstype
from mindspore import nn

from src.cell import WithLossCellD, WithLossCellG
from src.dcgan import DCGAN
from src.discriminator import Discriminator
from src.generator import Generator
from src.config import dcgan_imagenet_cfg as cfg

parser = argparse.ArgumentParser(description='ntsnet export')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file name.")
parser.add_argument('--data_url', default=None, help='Directory contains CUB_200_2011 dataset.')
parser.add_argument('--train_url', default=None, help='Directory contains checkpoint file')
parser.add_argument("--file_name", type=str, default="ntsnet", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", help="file format")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target (default: Ascend)')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    netD = Discriminator()
    netG = Generator()

    criterion = nn.BCELoss(reduction='mean')

    netD_with_criterion = WithLossCellD(netD, netG, criterion)
    netG_with_criterion = WithLossCellG(netD, netG, criterion)

    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)
    optimizerG = nn.Adam(netG.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)

    myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
    myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)

    net = DCGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
    param_dict = load_checkpoint(os.path.join(args.train_url, args.ckpt_file))
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # inputs = Tensor(np.random.rand(args.batch_size, 3, 448, 448), mstype.float32)
    real_data = Tensor(np.random.rand(args.batch_size, 3, 32, 32), mstype.float32)
    latent_code = Tensor(np.random.rand(args.batch_size, 100, 1, 1), mstype.float32)
    inputs = [real_data, latent_code]
    export(net, *inputs, file_name=args.file_name, file_format=args.file_format)
