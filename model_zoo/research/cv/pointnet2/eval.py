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

import numpy as np

import mindspore.dataset as ds
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import DatasetGenerator
from src.pointnet2 import PointNet2


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

    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)
    parser.add_argument('--eval_out_url', type=str)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    print('PARAMETER ...')
    args = parse_args()
    print(args)
    device_id = int(os.getenv('DEVICE_ID'))

    if args.modelarts:
        import moxing as mox

        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        pretrained_ckpt_path = "/cache/pretrained_ckpt/pretrained.ckpt"
        mox.file.copy_parallel(args.pretrained_ckpt, pretrained_ckpt_path)
        local_eval_url_out = "/cache/eval_out"
    else:
        local_data_url = args.data_path
        pretrained_ckpt_path = args.pretrained_ckpt

    if args.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        context.set_context(max_call_depth=2048)
        context.set_context(device_id=device_id)
    else:
        raise ValueError("Unsupported platform.")

    # DATA LOADING
    print('Load dataset ...')
    data_path = local_data_url

    test_dataset_generator = DatasetGenerator(root=data_path,
                                              args=args, split='test',
                                              process_data=args.process_data)
    test_dataset = ds.GeneratorDataset(test_dataset_generator,
                                       ["data", "label"],
                                       num_parallel_workers=16,
                                       shuffle=False)
    test_dataset = test_dataset.batch(batch_size=args.batch_size,
                                      drop_remainder=True,
                                      num_parallel_workers=16)

    steps_per_epoch = test_dataset.get_dataset_size()

    # MODEL LOADING
    num_class = args.num_category

    net = PointNet2(num_class, args.use_normals)

    # load checkpoint
    print("Load checkpoint: ", args.pretrained_ckpt)
    param_dict = load_checkpoint(pretrained_ckpt_path)
    load_param_into_net(net, param_dict)

    net.set_train(False)

    # Start Eval
    print('Starting eval ...')
    time_start = time.time()
    print('Time: ', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    acc = []
    total_correct = 0
    total_seen = 0
    for batch_id, data in enumerate(test_dataset.create_dict_iterator()):
        time_step = time.time()
        points = data['data']
        label = data['label']
        pred = net(points)
        pred_choice = np.argmax(pred.asnumpy(), axis=1)
        correct = (pred_choice == label.asnumpy()).sum()
        acc.append(correct / label.shape[0])
        total_correct += correct
        total_seen += label.shape[0]
        cost_time = time.time() - time_step
        print('batch: ', batch_id + 1, '/', steps_per_epoch,
              '\t| accuracy: ', "%.4f" % acc[-1],
              '\t| step_time: ', "%.2f" % cost_time, ' s')

    print("\nAccuracy: ", total_correct / total_seen)
    print("\nCheckPoint Path: ", args.pretrained_ckpt)
    print('\nEnd of eval.')
    print('\nTotal time cost: ', "%.2f" % ((time.time() - time_start) / 60), ' min')

    if args.modelarts == "True":
        mox.file.copy_parallel(local_eval_url_out, args.eval_url)
