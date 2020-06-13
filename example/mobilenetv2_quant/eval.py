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
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.mobilenetV2_quant import mobilenet_v2_quant
from src.dataset import create_dataset_py
from src.config import config_ascend

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--platform', type=str, default=None, help='run platform')
args_opt = parser.parse_args()

if __name__ == '__main__':
    config_platform = None
    net = None
    if args_opt.platform == "Ascend":
        config_platform = config_ascend
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=device_id, save_graphs=False)
        net = mobilenet_v2_quant(num_classes=config_platform.num_classes)
    else:
        raise ValueError("Unsupport platform.")

    loss = nn.SoftmaxCrossEntropyWithLogits(
        is_grad=False, sparse=True, reduction='mean')

    dataset = create_dataset_py(dataset_path=args_opt.dataset_path,
                                do_train=False,
                                config=config_platform,
                                platform=args_opt.platform,
                                batch_size=config_platform.batch_size)
    step_size = dataset.get_dataset_size()

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
