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
"""
#################train textcnn example on movie review########################
python train.py
"""
import argparse
import math

import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cfg_mr, cfg_subj, cfg_sst2
from src.textcnn import TextCNN
from src.textcnn import SoftmaxCrossEntropyExpand
from src.dataset import MovieReview, SST2, Subjectivity

parser = argparse.ArgumentParser(description='TextCNN')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_id', type=int, default=5, help='device id of GPU or Ascend.')
parser.add_argument('--dataset', type=str, default="MR", choices=['MR', 'SUBJ', 'SST2'])
args_opt = parser.parse_args()

if __name__ == '__main__':
    rank = 0
    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)
    if args_opt.dataset == 'MR':
        cfg = cfg_mr
        instance = MovieReview(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SUBJ':
        cfg = cfg_subj
        instance = Subjectivity(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SST2':
        cfg = cfg_sst2
        instance = SST2(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)

    dataset = instance.create_train_dataset(batch_size=cfg.batch_size, epoch_size=cfg.epoch_size)
    batch_num = dataset.get_dataset_size()

    base_lr = cfg.base_lr
    learning_rate = []
    warm_up = [base_lr / math.floor(cfg.epoch_size / 5) * (i + 1) for _ in range(batch_num) for i in
               range(math.floor(cfg.epoch_size / 5))]
    shrink = [base_lr / (16 * (i + 1)) for _ in range(batch_num) for i in range(math.floor(cfg.epoch_size * 3 / 5))]
    normal_run = [base_lr for _ in range(batch_num) for i in
                  range(cfg.epoch_size - math.floor(cfg.epoch_size / 5) - math.floor(cfg.epoch_size * 2 / 5))]
    learning_rate = learning_rate + warm_up + normal_run + shrink

    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=cfg.word_len,
                  num_classes=cfg.num_classes, vec_length=cfg.vec_length)
    # Continue training if set pre_trained to be True
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net, param_dict)

    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=learning_rate, weight_decay=cfg.weight_decay)
    loss = SoftmaxCrossEntropyExpand(sparse=True)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc': Accuracy()})

    config_ck = CheckpointConfig(save_checkpoint_steps=int(cfg.epoch_size*batch_num/2),
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="train_textcnn", directory=ckpt_save_dir, config=config_ck)
    loss_cb = LossMonitor()
    model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("train success")
