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
import time
import os
from mindspore import context, nn
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.Xception import xception
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth
from src.model_utils.config import config as args_opt, config_gpu, config_ascend
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num

set_seed(1)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, args_opt.modelarts_dataset_unzip_name)):
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

    if args_opt.modelarts_dataset_unzip_name:
        zip_file_1 = os.path.join(args_opt.data_path, args_opt.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(args_opt.data_path)

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
    args_opt.test_data_dir = args_opt.data_path
    if args_opt.modelarts_dataset_unzip_name:
        args_opt.test_data_dir = os.path.join(args_opt.test_data_dir, args_opt.folder_name_under_zip_file)
    args_opt.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args_opt.checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    if args_opt.device_target == "Ascend":
        config = config_ascend
    elif args_opt.device_target == "GPU":
        config = config_gpu
    else:
        raise ValueError("Unsupported device_target.")

    context.set_context(device_id=args_opt.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)

    # create dataset
    dataset = create_dataset(args_opt.test_data_dir, do_train=False, batch_size=config.batch_size, device_num=1, rank=0)
    # step_size = dataset.get_dataset_size()

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
    res = model.eval(dataset, dataset_sink_mode=True)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)


if __name__ == '__main__':
    run_eval()
