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
"""test ShuffleNetV1"""
import argparse
import time
from mindspore import context, nn
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.shufflenetv1 import ShuffleNetV1 as shufflenetv1
from src.config import config
from src.dataset import create_dataset
from src.crossentropysmooth import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Checkpoint file path')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--model_size', type=str, default='2.0x', help='ShuffleNetV1 model size',
                        choices=['2.0x', '1.5x', '1.0x', '0.5x'])
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False,
                        device_id=args_opt.device_id)

    # create dataset
    dataset = create_dataset(args_opt.dataset_path, do_train=False, device_num=1, rank=0)
    step_size = dataset.get_dataset_size()

    # define net
    net = shufflenetv1(model_size=args_opt.model_size)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                              num_classes=config.num_classes)

    # define model
    eval_metrics = {'Loss': nn.Loss(), 'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top_5_Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)

    # eval model
    start_time = time.time()
    res = model.eval(dataset, dataset_sink_mode=True)
    log = "result:" + str(res) + ", ckpt:'" + args_opt.checkpoint_path + "', time: " + str(
        (time.time() - start_time) * 1000)
    print(log)
    filename = './eval_log.txt'
    with open(filename, 'a') as file_object:
        file_object.write(log + '\n')
