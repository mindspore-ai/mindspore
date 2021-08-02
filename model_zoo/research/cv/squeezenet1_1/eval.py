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
"""eval squeezenet."""
import os
import ast
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth
from src.squeezenet import SqueezeNet as squeezenet
from src.dataset import create_dataset_imagenet as create_dataset
from src.config import config

local_data_url = '/cache/data'
local_ckpt_url = '/cache/ckpt.ckpt'

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset.')
parser.add_argument('--net', type=str, default='squeezenet', help='Model.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--data_url', type=str, default="None", help='Datapath')
parser.add_argument('--train_url', type=str, default="None", help='Train output path')
args_opt = parser.parse_args()

set_seed(1)


if __name__ == '__main__':

    target = args_opt.device_target
    if args_opt.device_target != "Ascend":
        raise ValueError("Unsupported device target.")

    # init context
    device_id = os.getenv('DEVICE_ID')
    device_id = int(device_id) if device_id else 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        device_id=device_id)

    # create dataset
    if args_opt.run_cloudbrain:
        import moxing as mox
        mox.file.copy_parallel(args_opt.checkpoint_path, local_ckpt_url)
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=False,
                                 repeat_num=1,
                                 batch_size=config.batch_size,
                                 target=target,
                                 run_distribute=False)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=False,
                                 repeat_num=1,
                                 batch_size=config.batch_size,
                                 target=target,
                                 run_distribute=False)
    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    if args_opt.run_cloudbrain:
        param_dict = load_checkpoint(local_ckpt_url)
    else:
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
    print("result:", res, "ckpt=", local_ckpt_url)
