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
"""evaluate_imagenet"""
import time
import os

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inceptionv4 import Inceptionv4

import mindspore.nn as nn
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def modelarts_process():
    """ modelarts process """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print('Extract Start...')
                print('unzip file num: {}'.format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print('unzip percent: {}%'.format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print('cost time: {}min:{}s.'.format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print('Extract Done')
            else:
                print('This is not zip.')
        else:
            print('Zip has been extracted.')

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + '.zip')
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = '/tmp/unzip_sync.lock'

        # Each server contains 8 devices as most
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print('Zip file path: ', zip_file_1)
            print('Unzip file save dir: ', save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print('===Finish extract data synchronization===')
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print('Device: {}, Finish sync unzip data from {} to {}.'.format(get_device_id(), zip_file_1, save_dir_1))
        print('#' * 200, os.listdir(save_dir_1))
        print('#' * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

    config.dataset_path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.checkpoint_path = os.path.join(config.dataset_path, config.checkpoint_path)


DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}

@moxing_wrapper(pre_process=modelarts_process)
def inception_v4_eval():

    if config.platform == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    create_dataset = DS_DICT[config.ds_type]

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform)
    net = Inceptionv4(classes=config.num_classes)
    ckpt = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, ckpt)
    net.set_train(False)
    config.rank = 0
    config.group_size = 1
    dataset = create_dataset(dataset_path=config.dataset_path, do_train=False, cfg=config)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)
    print('=' * 20, 'Evalute start', '=' * 20)
    metrics = model.eval(dataset, dataset_sink_mode=config.ds_sink_mode)
    print("metric: ", metrics)


if __name__ == '__main__':
    if config.ds_type == 'imagenet':
        config.dataset_path = os.path.join(config.dataset_path, 'validation_preprocess')
    inception_v4_eval()
