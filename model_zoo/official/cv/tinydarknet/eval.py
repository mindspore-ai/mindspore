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
##############test tinydarknet example on cifar10#################
python eval.py
"""
import os
import time

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from src.dataset import create_dataset_imagenet, create_dataset_cifar
from src.tinydarknet import TinyDarkNet
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num

set_seed(1)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("Unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("Unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("Cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.modelarts_dataset_unzip_name:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_id % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(config.device_id, zip_file_1, save_dir_1))
    config.checkpoint_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.checkpoint_path)
    if not os.path.exists(config.checkpoint_path):
        raise ValueError("Check parameter 'checkpoint_path'. for more details, you can see README.md")
    config.val_data_dir = config.data_path

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    cfg = config
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target,
                        device_id=cfg.device_id)
    if config.dataset_name == "imagenet":
        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        dataset = create_dataset_imagenet(cfg.val_data_dir, 1, False)
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)
    elif config.dataset_name == "cifar10":
        dataset = create_dataset_cifar(dataset_path=config.val_data_dir,
                                       do_train=True,
                                       repeat_num=1,
                                       batch_size=config.batch_size,
                                       target=cfg.device_target)
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        raise ValueError("Dataset is not support.")

    net = TinyDarkNet(num_classes=cfg.num_classes)
    param_dict = load_checkpoint(cfg.checkpoint_path)
    print("Load checkpoint from [{}].".format(cfg.checkpoint_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    acc = model.eval(dataset)
    print("accuracy: ", acc)

if __name__ == '__main__':
    run_eval()
