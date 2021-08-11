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
""" modelarts start"""

import argparse
import ast
import os
from functools import partial

import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net
from src.model_utils.config import config
from src.unet_medical.unet_model import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.utils import UnetEval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')

    # 数据集目录
    parser.add_argument('-d', '--data_url', type=str, default='',
                        help='the training data')

    # 抽取出来的超参配置
    parser.add_argument('--img_size', type=ast.literal_eval, default=[96, 96],
                        help='image size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='epochs')
    parser.add_argument('--repeat', type=int, default=1, help='repeat')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--cross_valid_ind', type=int, default=1,
                        help='cross valid ind')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='number of channels')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
                        help='keep checkpoint max')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--dataset', type=str, default='Cell_nuclei',
                        choices=('Cell_nuclei', 'ISBI2012'),
                        help='dataset')
    parser.add_argument('--loss_scale', type=float, default=1024.0,
                        help='loss scale')
    parser.add_argument('--FixedLossScaleManager', type=float, default=1024.0,
                        help='loss scale')
    parser.add_argument('--eval_activate', type=str, default="Softmax",
                        choices=("Softmax", "Argmax"), help="eval activate")
    parser.add_argument('--eval_resize', type=ast.literal_eval, default=False,
                        choices=(True, False), help="eval resize")
    parser.add_argument('-t', '--run_distribute', type=ast.literal_eval,
                        default=False, help='Run distribute, default: false.')
    parser.add_argument("--run_eval", type=ast.literal_eval, default=False,
                        help="Run evaluation when training, default is False.")
    parser.add_argument("--save_best_ckpt", type=ast.literal_eval,
                        default=True,
                        help="Save best checkpoint when run_eval is True, "
                             "default is True.")
    parser.add_argument("--eval_start_epoch", type=int, default=0,
                        help="Evaluation start epoch when run_eval is True, "
                             "default is 0.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval when run_eval is True, "
                             "default is 1.")
    parser.add_argument("--eval_metrics", type=str, default="dice_coeff",
                        choices=("dice_coeff", "iou"),
                        help="Evaluation metrics when run_eval is True, "
                             "support [dice_coeff, iou], "
                             "default is dice_coeff.")
    parser.add_argument("--device_id", type=str, default="0", help="device id")
    return parser.parse_args()


def set_config(args):
    config.img_size = args.img_size
    config.lr = args.lr
    config.epochs = args.epochs
    config.repeat = args.repeat
    config.batchsize = args.batchsize
    config.cross_valid_ind = args.cross_valid_ind
    config.num_classes = args.num_classes
    config.num_channels = args.num_channels
    config.keep_checkpoint_max = args.keep_checkpoint_max
    config.weight_decay = args.weight_decay
    config.loss_scale = args.loss_scale
    config.FixedLossScaleManager = args.FixedLossScaleManager
    config.eval_activate = args.eval_activate
    config.eval_resize = args.eval_resize
    config.dataset = args.dataset


def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(args, ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels,
                          n_classes=config.num_classes)
    elif config.model_name == 'unet_nested':
        net = NestedUNet(in_channel=config.num_channels,
                         n_class=config.num_classes,
                         use_deconv=config.use_deconv,
                         use_bn=config.use_bn, use_ds=False)
    elif config.model_name == 'unet_simple':
        net = UNet(in_channel=config.num_channels, n_class=config.num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(config.model_name))
    # return a parameter dict for model
    param_dict = load_checkpoint(ckpt_file)
    # load the parameter into net
    load_param_into_net(net, param_dict)
    net = UnetEval(net)
    input_data = Tensor(np.ones(
        [1, config.num_channels, args.img_size[0],
         args.img_size[1]]).astype(np.float32))
    air_file_name = os.path.join(os.path.dirname(ckpt_file), config.model_name)
    print(f"Start exporting AIR, ckpt_file = {ckpt_file}, input_tensor_shape ="
          f" {input_data.shape}, air_file_path = {air_file_name}.air.")
    export(net, input_data, file_name=air_file_name, file_format='AIR')


def main():
    args = parse_args()
    print("Training setting:", args)
    set_config(args)
    os.environ["DEVICE_ID"] = args.device_id
    from train import train_net
    train_func = partial(train_net,
                         cross_valid_ind=args.cross_valid_ind,
                         epochs=args.epochs,
                         batch_size=args.batchsize,
                         lr=args.lr)
    try:
        import moxing as mox
        cache_data_url = config.data_path
        os.makedirs(cache_data_url, exist_ok=True)
        mox.file.copy_parallel(args.data_url, cache_data_url)
        args.data_url = cache_data_url
        train_func()
        ckpt_id = f"ckpt_{args.device_id}"
        ckpt_dir = os.path.join(config.output_path, config.checkpoint_path, ckpt_id)
        _export_air(args, ckpt_dir)
        mox.file.copy_parallel(ckpt_dir, args.train_url)
    except ModuleNotFoundError:
        train_func()


if __name__ == '__main__':
    main()
