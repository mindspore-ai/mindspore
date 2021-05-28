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
import argparse
import os
import shutil
import subprocess
import time

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from detect import detect_dataset
from src.east import EAST

parser = argparse.ArgumentParser('mindspore icdar eval')

# device related
parser.add_argument(
    '--device_target',
    type=str,
    default='Ascend',
    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument(
    '--device_num',
    type=int,
    default=5,
    help='device where the code will be implemented. (Default: Ascend)')

parser.add_argument(
    '--test_img_path',
    default='/data/icdar2015/Test/image/',
    type=str,
    help='Train dataset directory.')
parser.add_argument('--checkpoint_path', default='best.ckpt', type=str,
                    help='The ckpt file of ResNet. Default: "".')
args, _ = parser.parse_known_args()

context.set_context(
    mode=context.GRAPH_MODE,
    enable_auto_mixed_precision=True,
    device_target=args.device_target,
    save_graphs=False,
    device_id=args.device_num)


def eval_model(name, img_path, submit, save_flag=True):
    if os.path.exists(submit):
        shutil.rmtree(submit)
    os.mkdir(submit)
    network = EAST()
    param_dict = load_checkpoint(name)
    load_param_into_net(network, param_dict)
    network.set_train(True)

    start_time = time.time()
    detect_dataset(network, img_path, submit)
    os.chdir(submit)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput(
        'python ./evaluate/script.py -g=./evaluate/gt.zip -s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit)


if __name__ == '__main__':
    model_name = args.checkpoint_path
    test_img_path = args.test_img_path
    submit_path = './submit'
    eval_model(model_name, test_img_path, submit_path)
