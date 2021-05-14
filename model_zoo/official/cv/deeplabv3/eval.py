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
"""eval deeplabv3."""

import os
import argparse

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.nets import net_factory
from src.utils.eval_utils import BuildEvalNetwork, net_eval

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                    device_id=int(os.getenv('DEVICE_ID')))


def parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3 eval')

    # val data
    parser.add_argument('--data_root', type=str, default='', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--scales', type=float, action='append', help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
    parser.add_argument("--input_format", type=str, choices=["NCHW", "NHWC"], default="NCHW",
                        help="NCHW or NHWC")

    # model
    parser.add_argument('--model', type=str, default='deeplab_v3_s16', help='select model')
    parser.add_argument('--ckpt_path', type=str, default='', help='model to evaluate')
    args_space, _ = parser.parse_known_args()
    return args_space


if __name__ == '__main__':
    args = parse_args()
    # network
    if args.model == 'deeplab_v3_s16':
        network = net_factory.nets_map[args.model](args.num_classes, 16)
    elif args.model == 'deeplab_v3_s8':
        network = net_factory.nets_map[args.model](args.num_classes, 8)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))
    eval_net = BuildEvalNetwork(network, args.input_format)
    # load model
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    net_eval(args, eval_net)
