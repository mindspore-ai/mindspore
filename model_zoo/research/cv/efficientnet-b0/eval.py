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
"""train efficientnet."""
import os
import ast
import argparse
from mindspore import context, nn
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.models.effnet import EfficientNet
from src.config import config
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    # modelarts parameter
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    # Ascend parameter
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')

    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run distribute')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)

    if args_opt.run_modelarts:
        import moxing as mox

        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data/'
        local_train_url = '/cache/ckpt/'
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        mox.file.copy_parallel(args_opt.train_url, local_train_url)
    else:
        context.set_context(device_id=args_opt.device_id)

    # create dataset
    if args_opt.run_modelarts:
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=False,
                                 batch_size=config.batch_size)
        ckpt_path = local_train_url + 'Efficientnet_b0-rank0-350_625.ckpt'
        param_dict = load_checkpoint(ckpt_path)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=False,
                                 batch_size=config.batch_size)
        param_dict = load_checkpoint(args_opt.checkpoint_path)
    step_size = dataset.get_dataset_size()

    # define net
    net = EfficientNet(1, 1)

    # load checkpoint
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top_5_Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)

    # eval model
    res = model.eval(dataset)
    if args_opt.run_modelarts:
        print("result:", res, "ckpt=", local_data_url)
    else:
        print("result:", res, "ckpt=", args_opt.checkpoint_path)
