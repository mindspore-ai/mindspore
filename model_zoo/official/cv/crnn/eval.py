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
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.metric import CRNNAccuracy

set_seed(1)

parser = argparse.ArgumentParser(description="CRNN eval")
parser.add_argument("--dataset_path", type=str, default=None, help="Dataset, default is None.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="checkpoint file path, default is None")
parser.add_argument('--platform', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='Running platform, choose from Ascend, GPU, and default is Ascend.')
parser.add_argument('--model', type=str, default='lowcase', help="Model type, default is uppercase")
parser.add_argument('--dataset', type=str, default='synth', choices=['synth', 'ic03', 'ic13', 'svt', 'iiit5k'])
args_opt = parser.parse_args()

if args_opt.model == 'lowcase':
    from src.config import config1 as config
else:
    from src.config import config2 as config

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
if args_opt.platform == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

if __name__ == '__main__':
    config.batch_size = 1
    max_text_length = config.max_text_length
    input_size = config.input_size
    # create dataset
    dataset = create_dataset(name=args_opt.dataset,
                             dataset_path=args_opt.dataset_path,
                             batch_size=config.batch_size,
                             is_training=False,
                             config=config)
    step_size = dataset.get_dataset_size()
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config)
    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # define model
    model = Model(net, loss_fn=loss, metrics={'CRNNAccuracy': CRNNAccuracy(config)})
    # start evaluation
    res = model.eval(dataset, dataset_sink_mode=args_opt.platform == 'Ascend')
    print("result:", res, flush=True)
