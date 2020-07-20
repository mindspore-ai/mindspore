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
"""train resnet."""
import os
import random
import argparse
import numpy as np
from mindspore import context
from mindspore import dataset as de
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.crossentropy import CrossEntropy

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default=None, help='Resnet Model, either resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default=None, help='Dataset, either cifar10 or imagenet2012')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet

    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset
else:
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset import create_dataset3 as create_dataset

if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)

    # create dataset
    if args_opt.net == "resnet50":
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                                 target=target)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if args_opt.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
