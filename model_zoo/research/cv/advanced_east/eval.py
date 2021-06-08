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
#################eval advanced_east on dataset########################
"""
import argparse
import datetime
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.logger import get_logger
from src.predict import predict
from src.score import eval_pre_rec_f1
from src.config import config as cfg
from src.dataset import load_adEAST_dataset
from src.model import get_AdvancedEast_net, AdvancedEast


set_seed(1)


def parse_args():
    """parameters"""
    parser = argparse.ArgumentParser('adveast evaling')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: None)')

    # logging and checkpoint related
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=5, help='ckpt_interval')

    parser.add_argument('--ckpt', type=str, default='Epoch_C0-6_4500.ckpt', help='ckpt to load')
    parser.add_argument('--method', type=str, default='score', choices=['loss', 'score', 'pred'], help='evaluation')
    parser.add_argument('--path', type=str, help='image path of prediction')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args_opt = parser.parse_args()

    args_opt.data_dir = cfg.data_dir
    args_opt.batch_size = 8
    args_opt.train_image_dir_name = cfg.data_dir + cfg.train_image_dir_name
    args_opt.mindsrecord_train_file = cfg.mindsrecord_train_file
    args_opt.train_label_dir_name = cfg.data_dir + cfg.train_label_dir_name
    args_opt.mindsrecord_test_file = cfg.mindsrecord_test_file
    args_opt.results_dir = cfg.results_dir
    args_opt.val_fname = cfg.val_fname
    args_opt.pixel_threshold = cfg.pixel_threshold
    args_opt.max_predict_img_size = cfg.max_predict_img_size
    args_opt.last_model_name = cfg.last_model_name
    args_opt.saved_model_file_path = cfg.saved_model_file_path
    args_opt.is_train = False

    return args_opt


def eval_loss(eval_arg):
    """get network and init"""
    loss_net, train_net = get_AdvancedEast_net(eval_arg)
    print(os.path.join(eval_arg.saved_model_file_path, eval_arg.ckpt))
    load_param_into_net(train_net, load_checkpoint(os.path.join(eval_arg.saved_model_file_path, eval_arg.ckpt)))
    train_net.set_train(False)
    loss = 0
    idx = 0
    for item in dataset.create_tuple_iterator():
        loss += loss_net(item[0], item[1])
        idx += 1
    print(loss / idx)


def eval_score(eval_arg):
    """get network and init"""
    net = AdvancedEast(eval_arg)
    load_param_into_net(net, load_checkpoint(os.path.join(eval_arg.saved_model_file_path, eval_arg.ckpt)))
    net.set_train(False)
    obj = eval_pre_rec_f1()
    with open(os.path.join(eval_arg.data_dir, eval_arg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()

    img_h, img_w = eval_arg.max_predict_img_size, eval_arg.max_predict_img_size
    x = np.zeros((eval_arg.batch_size, 3, img_h, img_w), dtype=np.float32)
    batch_list = np.arange(0, len(f_list), eval_arg.batch_size)
    for idx in tqdm(batch_list):
        gt_list = []
        for i in range(idx, min(idx + eval_arg.batch_size, len(f_list))):
            item = f_list[i]
            img_filename = str(item).strip().split(',')[0][:-4]
            img_path = os.path.join(eval_arg.train_image_dir_name, img_filename) + '.jpg'

            img = Image.open(img_path)
            img = img.resize((img_w, img_h), Image.NEAREST).convert('RGB')
            img = np.asarray(img)
            img = img / 1.
            mean = np.array((123.68, 116.779, 103.939)).reshape([1, 1, 3])
            img = ((img - mean)).astype(np.float32)
            img = img.transpose((2, 0, 1))
            x[i - idx] = img

            gt_list.append(np.load(os.path.join(eval_arg.train_label_dir_name, img_filename) + '.npy'))
        if idx + eval_arg.batch_size >= len(f_list):
            x = x[:len(f_list) - idx]
        y = net(Tensor(x))
        obj.add(y, gt_list)

    print(obj.val())


def pred(eval_arg):
    """pred"""
    img_path = eval_arg.path
    net = AdvancedEast(eval_arg)
    load_param_into_net(net, load_checkpoint(os.path.join(eval_arg.saved_model_file_path, eval_arg.ckpt)))
    predict(net, img_path, eval_arg.pixel_threshold)


if __name__ == '__main__':
    args = parse_args()

    device_num = int(os.environ.get("DEVICE_NUM", 1))
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
            context.set_context(device_id=args.device_id)
        elif args.device_target == "GPU":
            init()

        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    dataset, batch_num = load_adEAST_dataset(os.path.join(args.data_dir,
                                                          args.mindsrecord_test_file),
                                             batch_size=args.batch_size,
                                             device_num=device_num, rank_id=args.rank, is_training=False,
                                             num_parallel_workers=device_num)

    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')

    method_dict = {'loss': eval_loss, 'score': eval_score, 'pred': pred}

    method_dict[args.method](args)
