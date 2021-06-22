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

"""get args"""
import ast
import argparse

def get_args(phase):
    """Define the common options that are used in training."""
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--device_target', default='Ascend', help='enables npu')
    parser.add_argument('--device_id', type=int, default=0)

    if phase == 'train':
        parser.add_argument('--dataset', default='lsun', help='cifar10 | lsun')
        parser.add_argument('--dataroot', default=None, help='path to dataset')
        parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='train in Modelarts or not')
        parser.add_argument('--data_url', default=None, help='Location of data.')
        parser.add_argument('--train_url', default=None, help='Location of training outputs.')

        parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
        parser.add_argument('--imageSize', type=int, default=64, help='the height/width of the input image to network')
        parser.add_argument('--nc', type=int, default=3, help='input image channels')
        parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
        parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
        parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        parser.add_argument('--netD', default='', help="path to netD (to continue training)")
        parser.add_argument('--clamp_lower', type=float, default=-0.01)
        parser.add_argument('--clamp_upper', type=float, default=0.01)
        parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
        parser.add_argument('--noBN', type=ast.literal_eval, default=False, help='use batchnorm or not (for DCGAN)')
        parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
        parser.add_argument('--experiment', default=None, help='Where to store samples and models')
        parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

    elif phase == 'export':
        parser.add_argument('--config', required=True, type=str, help='path to generator config .json file')
        parser.add_argument('--ckpt_file', type=str, required=True, help="Checkpoint file path.")
        parser.add_argument('--file_name', type=str, default="WGAN", help="output file name prefix.")
        parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], \
                                                      default='AIR', help='file format')
        parser.add_argument('--nimages', required=True, type=int, help="number of images to generate", default=1)

    elif phase == 'eval':
        parser.add_argument('--config', required=True, type=str, help='path to generator config .json file')
        parser.add_argument('--ckpt_file', required=True, type=str, help='path to generator weights .ckpt file')
        parser.add_argument('--output_dir', required=True, type=str, help="path to output directory")
        parser.add_argument('--nimages', required=True, type=int, help="number of images to generate", default=1)

    elif phase == 'pre310':
        parser.add_argument('--config', required=True, type=str, help='path to generator config .json file')
        parser.add_argument('--pre_result_path', type=str, help="preprocess dir", default='./preprocess_Result/')
        parser.add_argument('--nimages', required=True, type=int, help="number of images to generate", default=1)

    elif phase == 'post310':
        parser.add_argument('--config', required=True, type=str, help='path to generator config .json file')
        parser.add_argument('--output_dir', type=str, help="path to output directory", default='./infer_output')
        parser.add_argument('--post_result_path', type=str, help="postprocess dir", default='./result_Files')
        parser.add_argument('--nimages', required=True, type=int, help="number of images to generate", default=1)

    args_opt = parser.parse_args()
    return args_opt
