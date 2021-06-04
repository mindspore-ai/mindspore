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
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(1)

if config.net_name == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset

@moxing_wrapper()
def eval_net():
    """eval net """
    target = config.device_target

    # init context
    device_id = os.getenv('DEVICE_ID')
    device_id = int(device_id) if device_id else 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path,
                             do_train=False,
                             batch_size=config.batch_size,
                             target=target)

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    if config.dataset == "imagenet":
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
    print("result:", res, "ckpt=", config.checkpoint_file_path)

if __name__ == '__main__':
    eval_net()
