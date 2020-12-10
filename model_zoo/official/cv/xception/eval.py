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
"""eval Xception."""
import argparse
from mindspore import context, nn
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.Xception import xception
from src.config import config
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    args_opt = parser.parse_args()

    context.set_context(device_id=args_opt.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)

    # create dataset
    dataset = create_dataset(args_opt.dataset_path, do_train=False, batch_size=config.batch_size, device_num=1, rank=0)
    step_size = dataset.get_dataset_size()

    # define net
    net = xception(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top_5_Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)

    # eval model
    res = model.eval(dataset, dataset_sink_mode=False)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
