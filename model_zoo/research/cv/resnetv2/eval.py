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
""" eval.py """
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default='resnetv2_50',
                    help='Resnetv2 Model, resnetv2_50, resnetv2_101, resnetv2_152')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset, cifar10, cifar100, imagenet2012')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--dataset_path', type=str, default="../CIFAR-10/cifar-10-verify-bin",
                    help='Dataset path.')
parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/train_resnetv2_cifar10-100_1562.ckpt",
                    help='Checkpoint file path.')
args_opt = parser.parse_args()

# import net
if args_opt.net == "resnetv2_50":
    from src.resnetv2 import PreActResNet50 as resnetv2
elif args_opt.net == 'resnetv2_101':
    from src.resnetv2 import PreActResNet101 as resnetv2
elif args_opt.net == 'resnetv2_152':
    from src.resnetv2 import PreActResNet152 as resnetv2

# import dataset
if args_opt.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
elif args_opt.dataset == "cifar100":
    from src.dataset import create_dataset2 as create_dataset
elif args_opt.dataset == 'imagenet2012':
    raise Exception("ImageNet2012 dataset not yet supported")

# import config
if args_opt.net == "resnetv2_50" or args_opt.net == "resnetv2_101" or args_opt.net == "resnetv2_152":
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
    elif args_opt.dataset == 'cifar100':
        from src.config import config2 as config

set_seed(1)

try:
    device_id = int(os.getenv('DEVICE_ID'))
except TypeError:
    device_id = 1
context.set_context(device_id=device_id)

if __name__ == '__main__':
    print("============== Starting Evaluating ==============")
    print(f"start evaluating {args_opt.net} on device {device_id}")

    # init context
    target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnetv2(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if args_opt.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
