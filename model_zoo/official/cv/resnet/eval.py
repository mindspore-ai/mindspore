# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(1)

if config.net_name in ("resnet18", "resnet34", "resnet50"):
    if config.net_name == "resnet18":
        from src.resnet import resnet18 as resnet
    if config.net_name == "resnet34":
        from src.resnet import resnet34 as resnet
    if config.net_name == "resnet50":
        from src.resnet import resnet50 as resnet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.dataset import create_dataset2 as create_dataset
elif config.net_name == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.dataset import create_dataset4 as create_dataset

@moxing_wrapper()
def eval_net():
    """eval net"""
    target = config.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             target=target)

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if config.dataset == "imagenet2012":
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
    print("result:", res, "ckpt=", config.checkpoint_file_path)

if __name__ == '__main__':
    eval_net()
