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

"""Train retinanet and get checkpoint files."""

import os
import ast
import time
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.retinanet import retinanetWithLossCell, TrainingWrapper, retinanet50, resnet50
from src.dataset import create_retinanet_dataset
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id


set_seed(1)


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("lr:[{:8.6f}]".format(self.lr_init[cb_params.cur_step_num-1]), flush=True)


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
def main():

    config.lr_init = ast.literal_eval(config.lr_init)
    config.lr_end_rate = ast.literal_eval(config.lr_end_rate)

    if config.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        if config.distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=get_device_id())
            init()
            device_num = get_device_num()
            rank = get_rank_id()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=get_device_id())

    else:
        raise ValueError("Unsupported platform.")

    mindrecord_file = os.path.join(config.mindrecord_dir, "retinanet.mindrecord0")

    loss_scale = float(config.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as retinanet.mindrecord0.
    dataset = create_retinanet_dataset(mindrecord_file, repeat_num=1,
                                       num_parallel_workers=config.workers,
                                       batch_size=config.batch_size, device_num=device_num, rank=rank)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    backbone = resnet50(config.num_classes)
    retinanet = retinanet50(backbone, config)
    net = retinanetWithLossCell(retinanet, config)
    init_net_param(net)

    if config.pre_trained:
        if config.pre_trained_epoch_size <= 0:
            raise KeyError("pre_trained_epoch_size must be greater than 0.")
        param_dict = load_checkpoint(config.pre_trained)
        if config.filter_weight:
            filter_checkpoint_parameter(param_dict)
        load_param_into_net(net, param_dict)

    lr = Tensor(get_lr(global_step=config.global_step,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                       warmup_epochs1=config.warmup_epochs1, warmup_epochs2=config.warmup_epochs2,
                       warmup_epochs3=config.warmup_epochs3, warmup_epochs4=config.warmup_epochs4,
                       warmup_epochs5=config.warmup_epochs5, total_epochs=config.epoch_size,
                       steps_per_epoch=dataset_size))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      config.momentum, config.weight_decay, loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    model = Model(net)
    print("Start train retinanet, the first epoch will be slower because of the graph compilation.")
    cb = [TimeMonitor(), LossMonitor()]
    cb += [Monitor(lr_init=lr.asnumpy())]
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="retinanet", directory=config.save_checkpoint_path, config=config_ck)
    if config.distribute:
        if rank == 0:
            cb += [ckpt_cb]
        model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    else:
        cb += [ckpt_cb]
        model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    main()
