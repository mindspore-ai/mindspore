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
"""Eval"""

import argparse
import ast
import os
import time

import mindspore.dataset as ds
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import DatasetGenerator
from src.pointnet2 import PointNet2, NLLLoss


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('MindSpore PointNet++ Eval Configurations.')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--data_path', type=str, default='../data/modelnet40_normal_resampled/', help='data path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', type=ast.literal_eval, default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--platform', type=str, default='Ascend', help='run platform')
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)

    return parser.parse_known_args()[0]


def run_eval():
    """Run eval"""
    args = parse_args()

    # INIT
    device_id = int(os.getenv('DEVICE_ID', '0'))

    if args.enable_modelarts:
        import moxing as mox

        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        pretrained_ckpt_path = "/cache/pretrained_ckpt/pretrained.ckpt"
        mox.file.copy_parallel(args.pretrained_ckpt, pretrained_ckpt_path)
        local_eval_url = "/cache/eval_out"
        mox.file.copy_parallel(args.train_url, local_eval_url)
    else:
        local_data_url = args.data_path
        pretrained_ckpt_path = args.pretrained_ckpt

    if args.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        context.set_context(max_call_depth=2048)
    else:
        raise ValueError("Unsupported platform.")

    print(args)

    # DATA LOADING
    print('Load dataset ...')
    data_path = local_data_url

    num_workers = 8
    test_ds_generator = DatasetGenerator(root=data_path, args=args, split='test', process_data=args.process_data)
    test_ds = ds.GeneratorDataset(test_ds_generator, ["data", "label"], num_parallel_workers=num_workers, shuffle=False)
    test_ds = test_ds.batch(batch_size=args.batch_size, drop_remainder=True, num_parallel_workers=num_workers)

    # MODEL LOADING
    net = PointNet2(args.num_category, args.use_normals)

    # load checkpoint
    print("Load checkpoint: ", args.pretrained_ckpt)
    param_dict = load_checkpoint(pretrained_ckpt_path)
    load_param_into_net(net, param_dict)

    net_loss = NLLLoss()

    model = Model(net, net_loss, metrics={"Accuracy": Accuracy()})

    # EVAL
    net.set_train(False)
    print('Starting eval ...')
    time_start = time.time()
    print('Time: ', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    result = model.eval(test_ds, dataset_sink_mode=True)
    print("result : {}".format(result))

    # END
    print('Total time cost: {} min'.format("%.2f" % ((time.time() - time_start) / 60)))

    if args.enable_modelarts:
        mox.file.copy_parallel(local_eval_url, args.train_url)


if __name__ == '__main__':
    run_eval()
