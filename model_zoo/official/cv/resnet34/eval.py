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
##############test resnet34 example on imagenet2012#################
python eval.py
"""
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.cross_entropy_smooth import CrossEntropySmooth

from src.resnet import resnet34 as resnet
from src.config import config
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--modelart', type=str, default=None, help='use modelart or not')
parser.add_argument('--ckpt_url', type=str, default=None, help='location of ckpt')
parser.add_argument('--train_url', type=str, default=None, help='Location of train log')
parser.add_argument('--data_url', type=str, default=None, help='Dataset imagenet2012')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
args_opt = parser.parse_args()

set_seed(1)

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

if __name__ == '__main__':

    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=int(os.environ["DEVICE_ID"]))

    # create dataset
    if args_opt.modelart:
        import moxing as mox
        data_path = '/cache/data_path'
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=data_path)
        tar_command = "tar -xvf /cache/data_path/imagenet_original.tar.gz -C /cache/data_path/"
        os.system(tar_command)
        data_path = '/cache/data_path/imagenet_original/'
    else:
        data_path = args_opt.data_url
    data_path = os.path.join(data_path, 'val')
    dataset = create_dataset(dataset_path=data_path, do_train=False, batch_size=config.batch_size)

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    if args_opt.modelart:
        import moxing as mox
        ckpt_path = '/cache/ckpt_path/'
        mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url=ckpt_path)
        param_dict = load_checkpoint("/cache/ckpt_path/resnet-90_625.ckpt")
    else:
        param_dict = args_opt.checkpoint_path
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    result = model.eval(dataset)
    print("result:", result)
