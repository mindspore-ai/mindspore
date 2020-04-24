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
eval.
"""
import os
import argparse
from dataset import create_dataset
from config import config
from mindspore import context
from mindspore.model_zoo.mobilenet import mobilenet_v2
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id, save_graphs=False)
context.set_context(enable_task_sink=True)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)

if __name__ == '__main__':
    context.set_context(enable_hccl=False)

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    net = mobilenet_v2()

    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
