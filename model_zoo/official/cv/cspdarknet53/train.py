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
"""train scripts."""
import os
import time
import datetime

from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.loss import CrossEntropy
from src.lr_generator import get_lr
from src.utils.logging import get_logger
from src.utils.optimizers_init import get_param_groups
from src.utils.var_init import load_pretrain_model
from src.image_classification import CSPDarknet53

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num


set_seed(1)


class ProgressMonitor(Callback):
    """monitor loss and cost time."""
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        cur_step_num = cb_params.cur_step_num - 1
        _epoch = cb_params.cur_epoch_num
        time_cost = time.time() - self.me_epoch_start_time
        fps_mean = self.args.per_batch_size * (cur_step_num - self.me_epoch_start_step_num) * \
                   self.args.group_size / time_cost
        per_step_time = 1000 * time_cost / (cur_step_num - self.me_epoch_start_step_num)
        self.args.logger.info('epoch[{}], iter[{}], loss: {}, mean_fps: {:.2f}'
                              ' imgs/sec, per_step_time: {:.2f} ms'.format(_epoch,
                                                                           cur_step_num % self.args.steps_per_epoch,
                                                                           cb_params.net_outputs,
                                                                           fps_mean,
                                                                           per_step_time))

        self.me_epoch_start_step_num = cur_step_num
        self.me_epoch_start_time = time.time()


def set_default_args(args):
    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.image_size = list(map(int, args.image_size.split(',')))

    args.rank = get_rank_id()
    args.group_size = get_device_num()

    args.group_size = get_device_num()
    if args.is_dynamic_loss_scale == 1:
        args.loss_scale = 1

    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S"))
    args.logger = get_logger(args.outputs_dir, args.rank)
    return args


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_config.modelarts_dataset_unzip_name)):
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

    if default_config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_config.data_path, default_config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_config.data_path)

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

    default_config.ckpt_path = os.path.join(default_config.output_path, default_config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    config = set_default_args(default_config)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False, device_id=get_device_id())
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                          gradients_mean=True)
        init()

    # dataloader
    de_dataset = create_dataset(config.data_dir, config.image_size, config.per_batch_size,
                                config.rank, config.group_size, num_parallel_workers=8)
    config.steps_per_epoch = de_dataset.get_dataset_size()

    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    network = CSPDarknet53(num_classes=config.num_classes)
    load_pretrain_model(config.pretrained, network, config)

    # lr
    lr = get_lr(config)

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)

    # loss
    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager,
                  metrics={'acc'}, amp_level="O2")

    # checkpoint save
    progress_cb = ProgressMonitor(config)
    callbacks = [progress_cb,]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
                                       keep_checkpoint_max=config.ckpt_save_max)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=True)



if __name__ == '__main__':
    run_train()
