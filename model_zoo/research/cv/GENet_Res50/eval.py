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
"""train GENet."""
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth
from src.GENet import GE_resnet50 as Net
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--extra', type=str, default="False",
                    help='whether to use Depth-wise conv to down sample')
parser.add_argument('--mlp', type=str, default="True",
                    help='bottleneck . whether to use 1*1 conv')
parser.add_argument('--is_modelarts', type=str, default="False", help='is train on modelarts')
args_opt = parser.parse_args()

if args_opt.extra.lower() == "false":
    from src.config import config3 as config
else:
    if args_opt.mlp.lower() == "false":
        from src.config import config2 as config
    else:
        from src.config import config1 as config

if args_opt.is_modelarts == "True":
    import moxing as mox

set_seed(1)

def trans_char_to_bool(str_):
    """
    Args:
        str_: string

    Returns:
        bool
    """
    result = False
    if str_.lower() == "true":
        result = True
    return result

if __name__ == '__main__':
    target = args_opt.device_target
    local_data_url = args_opt.data_url
    local_pretrained_url = args_opt.checkpoint_path

    if args_opt.is_modelarts == "True":
        local_data_url = "/cache/data"
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        local_pretrained_path = "/cache/pretrained"
        mox.file.make_dirs(local_pretrained_path)
        filename = "pretrained.ckpt"
        local_pretrained_url = os.path.join(local_pretrained_path, filename)
        mox.file.copy(args_opt.checkpoint_path, local_pretrained_url)

    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        save_graphs=False)

    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=local_data_url,
                             do_train=False,
                             batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    mlp = trans_char_to_bool(args_opt.mlp)
    extra = trans_char_to_bool(args_opt.extra)
    # define net
    net = Net(class_num=config.class_num, extra=extra, mlp=mlp)

    # load checkpoint
    param_dict = load_checkpoint(local_pretrained_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True,
                              reduction='mean',
                              smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
