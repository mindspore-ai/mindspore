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
"""Warpctc evaluation"""
import os
import math as m
import random
import argparse
import numpy as np
from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.loss import CTCLoss
from src.config import config as cf
from src.dataset import create_dataset
from src.warpctc import StackedRNN
from src.metric import WarpCTCAccuracy

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description="Warpctc training")
parser.add_argument("--dataset_path", type=str, default=None, help="Dataset, default is None.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="checkpoint file path, default is None")
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    save_graphs=False,
                    device_id=device_id)

if __name__ == '__main__':
    max_captcha_digits = cf.max_captcha_digits
    input_size = m.ceil(cf.captcha_height / 64) * 64 * 3
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, repeat_num=1, batch_size=cf.batch_size)
    step_size = dataset.get_dataset_size()
    # define loss
    loss = CTCLoss(max_sequence_length=cf.captcha_width, max_label_length=max_captcha_digits, batch_size=cf.batch_size)
    # define net
    net = StackedRNN(input_size=input_size, batch_size=cf.batch_size, hidden_size=cf.hidden_size)
    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # define model
    model = Model(net, loss_fn=loss, metrics={'WarpCTCAccuracy': WarpCTCAccuracy()})
    # start evaluation
    res = model.eval(dataset)
    print("result:", res, flush=True)
