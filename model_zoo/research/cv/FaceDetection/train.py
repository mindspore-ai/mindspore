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
"""Face detection train."""
import os
import time
import datetime
import numpy as np

from mindspore import context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import CheckpointConfig
from mindspore.common import dtype as mstype

from src.logging import get_logger
from src.data_preprocess import create_dataset
from src.network_define import define_network

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

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

    config.ckpt_path = os.path.join(config.output_path, "output")


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''train'''
    config.world_size = get_device_num()
    config.local_rank = get_rank_id()
    if config.run_platform == "CPU":
        config.use_loss_scale = False
        config.world_size = 1
        config.local_rank = 0
    if config.run_platform == "GPU":
        config.use_loss_scale = False
    if config.world_size != 8:
        config.lr_steps = [i * 8 // config.world_size for i in config.lr_steps]
    config.weight_decay = config.weight_decay if config.world_size != 1 else 0.
    config.outputs_dir = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    print('config.outputs_dir', config.outputs_dir)
    config.num_anchors_list = [len(x) for x in config.anchors_mask]
    print('=============yolov3 start trainging==================')
    devid = int(os.getenv('DEVICE_ID', '0')) if config.run_platform != 'CPU' else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform, save_graphs=False, device_id=devid)
    # init distributed
    if config.world_size != 1:
        init()
        config.local_rank = get_rank()
        config.world_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=config.world_size,
                                          gradients_mean=True)
    config.logger = get_logger(config.outputs_dir, config.local_rank)

    # dataloader
    ds = create_dataset(config)

    config.logger.important_info('start create network')
    create_network_start = time.time()

    train_net = define_network(config)

    # checkpoint
    ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
    train_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
    ckpt_cb = ModelCheckpoint(config=train_config, directory=config.outputs_dir, prefix='{}'.format(config.local_rank))
    cb_params = InternalCallbackParam()
    cb_params.train_network = train_net
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1
    i = 0
    if config.use_loss_scale:
        scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)
    for data in ds.create_tuple_iterator(output_numpy=True):
        batch_images = data[0]
        batch_labels = data[1]
        input_list = [Tensor(batch_images, mstype.float32)]
        for idx in range(2, 26):
            input_list.append(Tensor(data[idx], mstype.float32))
        if config.use_loss_scale:
            scaling_sens = Tensor(scale_manager.get_loss_scale(), dtype=mstype.float32)
            loss0, overflow, _ = train_net(*input_list, scaling_sens)
            overflow = np.all(overflow.asnumpy())
            if overflow:
                scale_manager.update_loss_scale(overflow)
            else:
                scale_manager.update_loss_scale(False)
            config.logger.info('rank[{:d}], iter[{}], loss[{}], overflow:{}, loss_scale:{}, lr:{}, batch_images:{}, '
                               'batch_labels:{}'.format(config.local_rank, i, loss0, overflow, scaling_sens,
                                                        config.lr[i], batch_images.shape, batch_labels.shape))
        else:
            loss0 = train_net(*input_list)
            config.logger.info('rank[{:d}], iter[{}], loss[{}], lr:{}, batch_images:{}, '
                               'batch_labels:{}'.format(config.local_rank, i, loss0,
                                                        config.lr[i], batch_images.shape, batch_labels.shape))
        # save ckpt
        cb_params.cur_step_num = i + 1  # current step number
        cb_params.batch_num = i + 2
        if config.local_rank == 0:
            ckpt_cb.step_end(run_context)

        # save Log
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            config.logger.important_info('Yolov3, graph compile time={:.2f}s'.format(time_for_graph_compile))

        if i % config.steps_per_epoch == 0:
            cb_params.cur_epoch_num += 1

        if i % config.log_interval == 0 and config.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / config.steps_per_epoch)
            fps = config.batch_size * (i - old_progress) * config.world_size / time_used
            config.logger.info('epoch[{}], iter[{}], loss:[{}], {:.2f} imgs/sec'.format(epoch, i, loss0, fps))
            t_end = time.time()
            old_progress = i

        if i % config.steps_per_epoch == 0 and config.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / config.steps_per_epoch)
            fps = config.batch_size * config.world_size * config.steps_per_epoch / epoch_time_used
            config.logger.info('=================================================')
            config.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            config.logger.info('=================================================')
            t_epoch = time.time()

        i = i + 1

    config.logger.info('=============yolov3 training finished==================')


if __name__ == "__main__":
    run_train()
