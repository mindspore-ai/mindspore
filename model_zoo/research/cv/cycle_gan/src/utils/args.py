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
"""get args."""
import argparse
import ast
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank

def get_args(phase):
    """Define the common options that are used in both training and test."""
    parser = argparse.ArgumentParser(description='Cycle GAN.')
    # basic parameters
    parser.add_argument('--model', type=str, default="resnet", choices=("resnet", "unet"), \
                        help='generator model, should be in [resnet, unet].')
    parser.add_argument('--platform', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"), \
                        help='run platform, only support GPU, CPU and Ascend')
    parser.add_argument("--device_id", type=int, default=0, help="device id, default is 0.")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate, default is 0.0002.")
    parser.add_argument('--pool_size', type=int, default=50, \
                        help='the size of image buffer that stores previously generated images, default is 50.')
    parser.add_argument('--lr_policy', type=str, default='linear', choices=("linear", "constant"), \
                        help='learning rate policy, default is linear')
    parser.add_argument("--image_size", type=int, default=256, help="input image_size, default is 256.")
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size, default is 1.')
    parser.add_argument('--max_epoch', type=int, default=200, help='epoch size for training, default is 200.')
    parser.add_argument('--n_epochs', type=int, default=100, \
                        help='number of epochs with the initial learning rate, default is 100')
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1, default is 0.5.")
    parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"), \
                        help='network initialization, default is normal.')
    parser.add_argument('--init_gain', type=float, default=0.02, \
                        help='scaling factor for normal, xavier and orthogonal, default is 0.02.')

    # model parameters
    parser.add_argument('--in_planes', type=int, default=3, help='input channels, default is 3.')
    parser.add_argument('--ngf', type=int, default=64, help='generator model filter numbers, default is 64.')
    parser.add_argument('--gl_num', type=int, default=9, help='generator model residual block numbers, default is 9.')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator model filter numbers, default is 64.')
    parser.add_argument('--dl_num', type=int, default=3, \
                        help='discriminator model residual block numbers, default is 3.')
    parser.add_argument('--slope', type=float, default=0.2, help='leakyrelu slope, default is 0.2.')
    parser.add_argument('--norm_mode', type=str, default="instance", choices=("batch", "instance"), \
                        help='norm mode, default is instance.')
    parser.add_argument('--lambda_A', type=float, default=10.0, \
                        help='weight for cycle loss (A -> B -> A), default is 10.')
    parser.add_argument('--lambda_B', type=float, default=10.0, \
                        help='weight for cycle loss (B -> A -> B), default is 10.')
    parser.add_argument('--lambda_idt', type=float, default=0.5, \
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the '
                             'weight of the identity mapping loss. For example, if the weight of the identity loss '
                             'should be 10 times smaller than the weight of the reconstruction loss,'
                             'please set lambda_identity = 0.1, default is 0.5.')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=("lsgan", "vanilla"), \
                        help='the type of GAN loss, default is lsgan.')
    parser.add_argument('--pad_mode', type=str, default='REFLECT', choices=("CONSTANT", "REFLECT", "SYMMETRIC"), \
                        help='the type of Pad, default is REFLECT.')
    parser.add_argument('--need_dropout', type=ast.literal_eval, default=True, \
                        help='whether need dropout, default is True.')

    # distillation learning parameters
    parser.add_argument('--kd', type=ast.literal_eval, default=False, \
                        help='knowledge distillation learning or not, default is False.')
    parser.add_argument('--t_ngf', type=int, default=64, \
                        help='teacher network generator model filter numbers when `kd` is True, default is 64.')
    parser.add_argument('--t_gl_num', type=int, default=9, \
                        help='teacher network generator model residual block numbers when `kd` is True, default is 9.')
    parser.add_argument('--t_slope', type=float, default=0.2, \
                        help='teacher network leakyrelu slope when `kd` is True, default is 0.2.')
    parser.add_argument('--t_norm_mode', type=str, default="instance", choices=("batch", "instance"), \
                        help='teacher network norm mode when `kd` is True, default is instance.')
    parser.add_argument("--GT_A_ckpt", type=str, default=None, \
                        help="teacher network pretrained checkpoint file path of G_A when `kd` is True.")
    parser.add_argument("--GT_B_ckpt", type=str, default=None, \
                        help="teacher network pretrained checkpoint file path of G_B when `kd` is True.")

    # additional parameters
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument("--G_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_A.")
    parser.add_argument("--G_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_B.")
    parser.add_argument("--D_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_A.")
    parser.add_argument("--D_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_B.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--print_iter", type=int, default=100, help="log print iter, default is 100.")
    parser.add_argument('--need_profiler', type=ast.literal_eval, default=False, \
                        help='whether need profiler, default is False.')
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False, \
                        help='whether save graphs, default is False.')
    parser.add_argument('--outputs_dir', type=str, default='./outputs', \
                        help='models are saved here, default is ./outputs.')
    parser.add_argument('--dataroot', default=None, \
                        help='path of images (should have subfolders trainA, trainB, testA, testB, etc).')
    parser.add_argument('--save_imgs', type=ast.literal_eval, default=True, \
                        help='whether save imgs when epoch end, if True result images will generate in '
                             '`outputs_dir/imgs`, default is True.')
    parser.add_argument('--use_random', type=ast.literal_eval, default=True, \
                        help='whether use random when training, default is True.')
    parser.add_argument('--max_dataset_size', type=int, default=None, help='max images pre epoch, default is None.')
    if phase == "export":
        parser.add_argument("--file_name", type=str, default="cyclegan", help="output file name prefix.")
        parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', \
                            help='file format')

    args = parser.parse_args()
    if args.device_num > 1 and args.platform != "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=args.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=args.device_num)
        init()
        args.rank = get_rank()
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform,
                            save_graphs=args.save_graphs, device_id=args.device_id)
        args.rank = 0
        args.device_num = 1

    if args.platform != "GPU":
        args.pad_mode = "CONSTANT"

    if phase != "train" and (args.G_A_ckpt is None or args.G_B_ckpt is None):
        raise ValueError('Must set G_A_ckpt and G_B_ckpt in predict phase!')

    if args.kd:
        if args.GT_A_ckpt is None or args.GT_B_ckpt is None:
            raise ValueError('Must set GT_A_ckpt, GT_B_ckpt in knowledge distillation!')

    if args.norm_mode == "instance" or (args.kd and args.t_norm_mode == "instance"):
        args.batch_size = 1

    if args.dataroot is None and (phase in ["train", "predict"]):
        raise ValueError('Must set dataroot!')

    if not args.use_random:
        args.need_dropout = False
        args.init_type = "constant"

    if args.max_dataset_size is None:
        args.max_dataset_size = float("inf")

    args.n_epochs = min(args.max_epoch, args.n_epochs)
    args.n_epochs_decay = args.max_epoch - args.n_epochs
    args.phase = phase
    return args
