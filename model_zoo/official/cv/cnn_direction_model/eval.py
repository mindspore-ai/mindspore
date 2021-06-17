# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""test direction model."""
import os
import time
import random
import numpy as np
from mindspore import context
from mindspore import dataset as de
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.cnn_direction_model import CNNDirectionModel
from src.dataset import create_dataset_eval
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.model_utils.config import config

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)


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
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_eval():
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    device_id = get_device_id()
    context.set_context(device_id=device_id)

    # create dataset
    dataset_name = config.dataset_name
    dataset_lr, dataset_rl = create_dataset_eval(config.eval_dataset_path + "/" + dataset_name +
                                                 ".mindrecord0", config=config, dataset_name=dataset_name)
    step_size = dataset_lr.get_dataset_size()

    print("step_size ", step_size)

    # define net
    net = CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512])

    # load checkpoint
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="sum")

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})

    # eval model
    res_lr = model.eval(dataset_lr, dataset_sink_mode=False)
    res_rl = model.eval(dataset_rl, dataset_sink_mode=False)
    print("result on upright images:", res_lr, "ckpt=", config.checkpoint_path)
    print("result on 180 degrees rotated images:", res_rl, "ckpt=", config.checkpoint_path)


if __name__ == '__main__':
    model_eval()
