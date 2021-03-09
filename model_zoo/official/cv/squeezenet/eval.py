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
"""eval squeezenet."""
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default='squeezenet', choices=['squeezenet', 'squeezenet_residual'],
                    help='Model.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'], help='Dataset.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()

set_seed(1)

if args_opt.net == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset_imagenet as create_dataset
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
    if args_opt.dataset == "cifar10":
        from src.config import config3 as config
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.config import config4 as config
        from src.dataset import create_dataset_imagenet as create_dataset

if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    device_id = os.getenv('DEVICE_ID')
    device_id = int(device_id) if device_id else 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=False,
                             batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    if args_opt.dataset == "imagenet":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True,
                                  reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net,
                  loss_fn=loss,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
