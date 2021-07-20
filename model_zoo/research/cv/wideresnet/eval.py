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
"""
##############test WideResNet example on cifar10#################
python eval.py
"""
import os
import ast
import argparse
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.cross_entropy_smooth import CrossEntropySmooth
from src.wide_resnet import wideresnet
from src.dataset import create_dataset
from src.config import config_WideResnet as cfg


parser = argparse.ArgumentParser(description='Ascend WideResNet CIFAR10 Eval')
parser.add_argument('--data_url', required=True, default=None, help='Location of data')
parser.add_argument('--ckpt_url', type=str, default=None, help='location of ckpt')
parser.add_argument('--modelart', required=True, type=ast.literal_eval, default=False,
                    help='training on modelart or not, default is False')
args = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

if __name__ == '__main__':

    target = 'Ascend'

    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=int(os.environ["DEVICE_ID"]))

    data_path = '/cache/data_path'

    if args.modelart:
        import moxing as mox
        mox.file.copy_parallel(src_url=args.data_url, dst_url=data_path)
    else:
        data_path = args.data_url

    ds_eval = create_dataset(dataset_path=data_path,
                             do_train=False,
                             repeat_num=cfg.repeat_num,
                             batch_size=cfg.batch_size)

    net = wideresnet()

    ckpt_path = '/cache/ckpt_path/'
    if args.modelart:
        import moxing as mox
        mox.file.copy_parallel(args.ckpt_url, dst_url=ckpt_path)
        param_dict = load_checkpoint('/cache/ckpt_path/WideResNet_best.ckpt')
    else:
        param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    if not cfg.use_label_smooth:
        cfg.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})

    output = model.eval(ds_eval)

    print("result:", output)
