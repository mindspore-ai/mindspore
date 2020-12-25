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
##############test textcnn example on movie review#################
python eval.py
"""
import argparse

import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cfg_mr, cfg_subj, cfg_sst2
from src.textcnn import TextCNN
from src.dataset import MovieReview, SST2, Subjectivity

parser = argparse.ArgumentParser(description='TextCNN')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset', type=str, default="MR", choices=['MR', 'SUBJ', 'SST2'])
args_opt = parser.parse_args()

if __name__ == '__main__':
    if args_opt.dataset == 'MR':
        cfg = cfg_mr
        instance = MovieReview(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SUBJ':
        cfg = cfg_subj
        instance = Subjectivity(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SST2':
        cfg = cfg_sst2
        instance = SST2(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    if device_target == "Ascend":
        context.set_context(device_id=cfg.device_id)
    dataset = instance.create_test_dataset(batch_size=cfg.batch_size)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=cfg.word_len,
                  num_classes=cfg.num_classes, vec_length=cfg.vec_length)
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.001,
                  weight_decay=cfg.weight_decay)

    if args_opt.checkpoint_path is not None:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        print("load checkpoint from [{}].".format(args_opt.checkpoint_path))
    else:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        print("load checkpoint from [{}].".format(cfg.checkpoint_path))

    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc': Accuracy()})

    acc = model.eval(dataset)
    print("accuracy: ", acc)
