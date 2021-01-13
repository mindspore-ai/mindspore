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
"""DPN model eval with MindSpore"""
import os
import argparse

from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dpn import dpns
from src.config import config
from src.imagenet_dataset import classification_dataset
set_seed(1)
# set context
device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend", save_graphs=False, device_id=device_id)


def parse_args():
    """parameters"""
    parser = argparse.ArgumentParser('dpn evaluating')
    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='eval data dir')
    # network related
    parser.add_argument('--pretrained', type=str, default='', help='ckpt path to load')
    args, _ = parser.parse_known_args()
    args.image_size = config.image_size
    args.num_classes = config.num_classes
    args.batch_size = config.batch_size
    args.num_parallel_workers = config.num_parallel_workers
    args.backbone = config.backbone
    args.loss_scale_num = config.loss_scale_num
    args.rank = config.rank
    args.group_size = config.group_size
    args.dataset = config.dataset
    return args


def dpn_evaluate(args):
    # create evaluate dataset
    eval_path = os.path.join(args.data_dir, 'val')
    eval_dataset = classification_dataset(eval_path,
                                          image_size=args.image_size,
                                          num_parallel_workers=args.num_parallel_workers,
                                          per_batch_size=args.batch_size,
                                          max_epoch=1,
                                          rank=args.rank,
                                          shuffle=False,
                                          group_size=args.group_size,
                                          mode='eval')

    # create network
    net = dpns[args.backbone](num_classes=args.num_classes)
    # load checkpoint
    load_param_into_net(net, load_checkpoint(args.pretrained))
    print("load checkpoint from [{}].".format(args.pretrained))
    # loss
    if args.dataset == "imagenet-1K":
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        if not args.label_smooth:
            args.label_smooth_factor = 0.0
        loss = CrossEntropy(smooth_factor=args.label_smooth_factor, num_classes=args.num_classes)

        # create model
    model = Model(net, amp_level="O2", keep_batchnorm_fp32=False, loss_fn=loss,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})
    # evaluate
    output = model.eval(eval_dataset)
    print(f'Evaluation result: {output}.')


if __name__ == '__main__':
    dpn_evaluate(parse_args())
    print('DPN evaluate success!')
