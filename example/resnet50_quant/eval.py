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
from src.dataset import create_dataset_py
from src.config import config
from src.crossentropy import CrossEntropy
from src.utils import _load_param_into_net
from models.resnet_quant import resnet50_quant
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()
target = args_opt.device_target
context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
if target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

if __name__ == '__main__':

    net = resnet50_quant(class_num=config.class_num)
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    if args_opt.do_eval:
        dataset = create_dataset_py(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                                    target=target)
        step_size = dataset.get_dataset_size()

        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            _load_param_into_net(net, param_dict)
        net.set_train(False)

        model = Model(net, loss_fn=loss, metrics={'acc'})
        res = model.eval(dataset)
        print("result:", res, "ckpt=", args_opt.checkpoint_path)
