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
get word2vec embeddings by running trian.py.
python train.py --device_target=[DEVICE_TARGET]
"""

import argparse
import ast
import os

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.config import w2v_cfg
from src.dataset import DataController
from src.lr_scheduler import poly_decay_lr
from src.skipgram import SkipGram

parser = argparse.ArgumentParser(description='Train SkipGram')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device target, support Ascend and GPU.')
parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
parser.add_argument('--pre_trained', type=str, default=None, help='the pretrained checkpoint file path.')
parser.add_argument('--train_data_dir', type=str, default=None, help='the directory of train data.')
args = parser.parse_args()
set_seed(1)

if __name__ == '__main__':
    if not os.path.exists(w2v_cfg.temp_dir):
        os.mkdir(w2v_cfg.temp_dir)
    if not os.path.exists(w2v_cfg.ckpt_dir):
        os.mkdir(w2v_cfg.ckpt_dir)

    print("Set Context...")
    rank_size = int(os.getenv('RANK_SIZE')) if args.run_distribute else 1
    rank_id = int(os.getenv('RANK_ID')) if args.run_distribute else 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        save_graphs=False)
    if args.run_distribute:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    print('Done.')

    print("Get Mindrecord...")
    train_data_dir = w2v_cfg.train_data_dir
    if args.train_data_dir:
        train_data_dir = args.train_data_dir

    data_controller = DataController(train_data_dir, w2v_cfg.ms_dir, w2v_cfg.min_count, w2v_cfg.window_size,
                                     w2v_cfg.neg_sample_num, w2v_cfg.data_epoch, w2v_cfg.batch_size,
                                     rank_size, rank_id)
    dataset = data_controller.get_mindrecord_dataset(col_list=['c_words', 'p_words', 'n_words'])
    print('Done.')

    print("Configure Training Parameters...")
    config_ck = CheckpointConfig(save_checkpoint_steps=w2v_cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=w2v_cfg.keep_checkpoint_max)
    ckpoint = ModelCheckpoint(prefix="w2v", directory=w2v_cfg.ckpt_dir, config=config_ck)
    loss_monitor = LossMonitor(1000)
    time_monitor = TimeMonitor()
    total_step = dataset.get_dataset_size() * w2v_cfg.train_epoch
    print('Total Step:', total_step)
    decay_step = min(total_step, int(2.4e6) // rank_size)
    lrs = Tensor(poly_decay_lr(w2v_cfg.lr, w2v_cfg.end_lr, decay_step, total_step, w2v_cfg.power,
                               update_decay_step=False))

    callbacks = [loss_monitor, time_monitor]
    if rank_id == 0:
        callbacks = [loss_monitor, time_monitor, ckpoint]

    net = SkipGram(data_controller.get_vocabs_size(), w2v_cfg.emb_size)
    if args.pre_trained:
        load_param_into_net(net, load_checkpoint(args.pre_trained))
    optim = nn.Adam(net.trainable_params(), learning_rate=lrs)
    train_net = nn.TrainOneStepCell(network=net, optimizer=optim)
    model = Model(train_net)
    print('Done.')

    print("Train Model...")
    model.train(epoch=w2v_cfg.train_epoch, train_dataset=dataset,
                callbacks=callbacks, dataset_sink_mode=w2v_cfg.dataset_sink_mode)
    print('Done.')

    print("Save Word2Vec Embedding...")
    net.save_w2v_emb(w2v_cfg.w2v_emb_save_dir, data_controller.id2word)  # save word2vec embedding
    print('Done.')

    print("End.")
