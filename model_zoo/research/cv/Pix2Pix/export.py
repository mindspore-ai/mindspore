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
    export checkpoint file into air, onnx, mindir models
"""
import argparse
import ast
import numpy as np
from mindspore import Tensor, nn, context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from src.models.pix2pix import Pix2Pix, get_generator, get_discriminator
from src.models.loss import D_Loss, D_WithLossCell, G_Loss, G_WithLossCell, TrainOneStepCell
from src.utils.tools import get_lr

parser = argparse.ArgumentParser(description='export')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default is 0.")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size, default is 1.")
parser.add_argument("--image_size", type=int, default=256, help="images size, default is 256.")
parser.add_argument('--ckpt_dir', type=str, default='./results/ckpt',
                    help='during training, the file path of stored CKPT.')
parser.add_argument("--ckpt", type=str, default=None, help="during validating, the file path of the CKPT used.")
parser.add_argument('--train_data_dir', type=str, default=None, help='the file path of input data during training.')
parser.add_argument("--file_name", type=str, default="Pix2Pix", help="output file name.")
parser.add_argument("--file_format", type=str, default="AIR", choices=["AIR", "ONNX", "MINDIR"], help="file format")
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                    help='device where the code will be implemented (default: Ascend)')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

if __name__ == '__main__':
    netG = get_generator()
    netD = get_discriminator()

    pix2pix = Pix2Pix(generator=netG, discriminator=netD)

    d_loss_fn = D_Loss()
    g_loss_fn = G_Loss()
    d_loss_net = D_WithLossCell(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = G_WithLossCell(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.netD.trainable_params(), learning_rate=get_lr(), beta1=0.5, beta2=0.999, loss_scale=1)
    g_opt = nn.Adam(pix2pix.netG.trainable_params(), learning_rate=get_lr(), beta1=0.5, beta2=0.999, loss_scale=1)

    train_net = TrainOneStepCell(loss_netD=d_loss_net, loss_netG=g_loss_net, optimizerD=d_opt, optimizerG=g_opt, sens=1)
    train_net.set_train()
    train_net = train_net.loss_netG

    ckpt_url = args.ckpt
    param_G = load_checkpoint(ckpt_url)
    load_param_into_net(netG, param_G)

    input_shp = [args.batch_size, 3, args.image_size, args.image_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    target_shp = [args.batch_size, 3, args.image_size, args.image_size]
    target_array = Tensor(np.random.uniform(-1.0, 1.0, size=target_shp).astype(np.float32))
    inputs = [input_array, target_array]
    file = f"{args.file_name}"
    export(train_net, *inputs, file_name=file, file_format=args.file_format)
