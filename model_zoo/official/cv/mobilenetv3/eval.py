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
import argparse
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.dataset import create_dataset_cifar
from src.config import config_gpu
from src.config import config_cpu
from src.mobilenetV3 import mobilenet_v3_large


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default="GPU", help='run device_target')
args_opt = parser.parse_args()


if __name__ == '__main__':
    config = None
    if args_opt.device_target == "GPU":
        config = config_gpu
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU", save_graphs=False)
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=False,
                                 config=config,
                                 device_target=args_opt.device_target,
                                 batch_size=config.batch_size)
    elif args_opt.device_target == "CPU":
        config = config_cpu
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="CPU", save_graphs=False)
        dataset = create_dataset_cifar(dataset_path=args_opt.dataset_path,
                                       do_train=False,
                                       batch_size=config.batch_size)
    else:
        raise ValueError("Unsupported device_target.")

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net = mobilenet_v3_large(num_classes=config.num_classes, activation="Softmax")

    step_size = dataset.get_dataset_size()

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
