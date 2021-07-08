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
################################eval glore_resnet50################################
python eval.py
"""
import os
import ast
import random
import argparse
import numpy as np

from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.glore_resnet50 import glore_resnet50
from src.dataset import create_eval_dataset
from src.loss import CrossEntropySmooth, SoftmaxCrossEntropyExpand
from src.config import config

parser = argparse.ArgumentParser(description='Image classification with glore_resnet50')
parser.add_argument('--use_glore', type=ast.literal_eval, default=True, help='Enable GloreUnit')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, help='Train output in modelarts')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--ckpt_url', type=str, default=None)
parser.add_argument('--is_modelarts', type=ast.literal_eval, default=True)
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
args_opt = parser.parse_args()

if args_opt.is_modelarts:
    import moxing as mox

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':
    target = args_opt.device_target
    # init context
    device_id = args_opt.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=device_id)

    # dataset
    eval_dataset_path = os.path.join(args_opt.data_url, 'val')
    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset')
        eval_dataset_path = '/cache/dataset/'
    predict_data = create_eval_dataset(dataset_path=eval_dataset_path, repeat_num=1, batch_size=config.batch_size)
    step_size = predict_data.get_dataset_size()
    if step_size == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    # define net
    net = glore_resnet50(class_num=config.class_num, use_glore=args_opt.use_glore)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.ckpt_url)
    load_param_into_net(net, param_dict)

    # define loss, model
    if config.use_label_smooth:
        loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyExpand(sparse=True)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    print("============== Starting Testing ==============")
    print("ckpt path : {}".format(args_opt.ckpt_url))
    print("data path : {}".format(eval_dataset_path))
    acc = model.eval(predict_data)
    print("==============Acc: {} ==============".format(acc))
