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
CRNN-Seq2Seq-OCR train.

"""

import os
import argparse
import datetime

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, LossMonitor, TimeMonitor

from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.dataset import create_ocr_train_dataset
from src.logger import get_logger
from src.attention_ocr import AttentionOCR, AttentionOCRWithLossCell, TrainingWrapper


set_seed(1)


def parse_args():
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore CRNN-Seq2Seq-OCR training')

    # device related
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="device where the code will be implemented.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0,
                        help='Distribute train or not, 1 for yes, 0 for no. Default: 0')
    parser.add_argument('--rank_id', type=int, default=0, help='Local rank of distributed. Default: 0')
    parser.add_argument('--device_num', type=int, default=1, help='World size of device. Default: 1')

    #dataset related
    parser.add_argument('--mindrecord_file', type=str, default='', help='Train dataset directory.')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval steps. Default: 100')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='Checkpoint save location. Default: outputs/')
    parser.add_argument('--pre_checkpoint_path', type=str, default='', help='Checkpoint save location.')
    parser.add_argument('--ckpt_interval', type=int, default=None, help='Save checkpoint interval. Default: None')

    parser.add_argument('--is_save_on_master', type=int, default=0,
                        help='Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 0')

    args, _ = parser.parse_known_args()

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    return args


def train():
    """Train function."""
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    if args.is_distributed:
        rank = args.rank_id
        device_num = args.device_num
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1

    # Logger
    args.logger = get_logger(args.outputs_dir, rank)
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # DATASET
    dataset = create_ocr_train_dataset(args.mindrecord_file,
                                       config.batch_size,
                                       rank_size=device_num,
                                       rank_id=rank)
    args.steps_per_epoch = dataset.get_dataset_size()
    args.logger.info('Finish loading dataset')

    if not args.ckpt_interval:
        args.ckpt_interval = args.steps_per_epoch
    args.logger.save_args(args)

    network = AttentionOCR(config.batch_size,
                           int(config.img_width / 4),
                           config.encoder_hidden_size,
                           config.decoder_hidden_size,
                           config.decoder_output_size,
                           config.max_length,
                           config.dropout_p)

    if args.pre_checkpoint_path:
        param_dict = load_checkpoint(args.pre_checkpoint_path)
        load_param_into_net(network, param_dict)

    network = AttentionOCRWithLossCell(network, config.max_length)

    lr = Tensor(config.lr, mstype.float32)
    opt = nn.Adam(network.trainable_params(), lr, beta1=config.adam_beta1, beta2=config.adam_beta2,
                  loss_scale=config.loss_scale)

    network = TrainingWrapper(network, opt, sens=config.loss_scale)

    args.logger.info('Finished get network')

    callback = [TimeMonitor(data_size=1), LossMonitor()]
    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.steps_per_epoch,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix="crnn_seq2seq_ocr")
        callback.append(ckpt_cb)

    model = Model(network)
    model.train(config.num_epochs, dataset, callbacks=callback, dataset_sink_mode=False)

    args.logger.info('==========Training Done===============')


if __name__ == "__main__":
    train()
