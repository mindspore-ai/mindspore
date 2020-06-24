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
import random
import numpy as np
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
from mindspore.communication.management import init
from src.resnet101 import resnet101
from src.dataset import create_dataset
from src.config import config
from src.crossentropy import CrossEntropy

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

if __name__ == '__main__':
    if not args_opt.do_eval and args_opt.run_distribute:
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True, parameter_broadcast=True)
        auto_parallel_context().set_all_reduce_fusion_split_indices([180, 313])
        init()

    epoch_size = config.epoch_size
    net = resnet101(class_num=config.class_num)

    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    if args_opt.do_eval:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size)
        step_size = dataset.get_dataset_size()

        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            load_param_into_net(net, param_dict)
        net.set_train(False)

        model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
        res = model.eval(dataset)
        print("result:", res, "ckpt=", args_opt.checkpoint_path)
