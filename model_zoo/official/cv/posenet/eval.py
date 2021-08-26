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
"""test posenet"""
import ast
import os
import time
import argparse
import math
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.config import common_config, KingsCollege, StMarysChurch
from src.posenet import PoseNet
from src.dataset import data_to_mindrecord, create_posenet_dataset

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='posenet eval')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--dataset', type=str, default='KingsCollege',
                        choices=['KingsCollege', 'StMarysChurch'],
                        help='dataset name.')
    parser.add_argument('--ckpt_url', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='Train in Modelarts.')
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'GPU'],
                        help='Name of device target.')
    args_opt = parser.parse_args()

    cfg = common_config
    if args_opt.dataset == "KingsCollege":
        dataset_cfg = KingsCollege
    elif args_opt.dataset == "StMarysChurch":
        dataset_cfg = StMarysChurch

    device_target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    if args_opt.device_id is not None:
        context.set_context(device_id=args_opt.device_id)
    else:
        context.set_context(device_id=cfg.device_id)

    eval_dataset_path = dataset_cfg.dataset_path
    if args_opt.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args_opt.data_url,
                               dst_url='/cache/dataset_eval/device_' + os.getenv('DEVICE_ID'))
        eval_dataset_path = '/cache/dataset_eval/device_' + os.getenv('DEVICE_ID') + '/'

    # It will generate eval mindrecord file in cfg.mindrecord_dir,
    # and the file name is "dataset_cfg.name + _posenet_eval.mindrecord".
    prefix = "_posenet_eval.mindrecord"
    mindrecord_dir = dataset_cfg.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, dataset_cfg.name + prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for eval.")
        data_to_mindrecord(eval_dataset_path, False, mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    dataset = create_posenet_dataset(mindrecord_file, batch_size=1, device_num=1, is_training=False)
    data_num = dataset.get_dataset_size()

    net = PoseNet()
    param_dict = load_checkpoint(args_opt.ckpt_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    print("Processing, please wait a moment.")
    results = np.zeros((data_num, 2))
    for step, item in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        image = item['image']
        poses = item['image_pose']

        pose_x = np.squeeze(poses[:, 0:3])
        pose_q = np.squeeze(poses[:, 3:])
        p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = net(Tensor(image))
        predicted_x = p3_x.asnumpy()
        predicted_q = p3_q.asnumpy()

        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi
        error_x = np.linalg.norm(pose_x - predicted_x)

        results[step, :] = [error_x, theta]
        print('Iteration:  ', step, ', Error XYZ (m):  ', error_x, ', Error Q (degrees):  ', theta)

    median_result = np.median(results, axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')
