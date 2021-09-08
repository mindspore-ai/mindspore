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
"""train Xception."""
import os
import time

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import dtype as mstype
from mindspore.common import set_seed

from src.lr_generator import get_lr
from src.Xception import xception
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth
from src.model_utils.config import config as args_opt, config_gpu, config_ascend
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num
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
    args_opt.train_data_dir = args_opt.data_path
    if args_opt.modelarts_dataset_unzip_name:
        args_opt.train_data_dir = os.path.join(args_opt.train_data_dir, args_opt.folder_name_under_zip_file)
    config_gpu.save_checkpoint_path = os.path.join(args_opt.output_path, config_gpu.save_checkpoint_path)
    config_ascend.save_checkpoint_path = os.path.join(args_opt.output_path, config_ascend.save_checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    if args_opt.device_target == "Ascend":
        config = config_ascend
    elif args_opt.device_target == "GPU":
        config = config_gpu
    else:
        raise ValueError("Unsupported device_target.")

    # init distributed
    if args_opt.is_distributed:
        context.set_context(device_id=get_device_id(), mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                            save_graphs=False)
        init()
        rank = get_rank_id()
        group_size = get_device_num()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=group_size, gradients_mean=True)
    else:
        rank = 0
        group_size = 1
        device_id = get_device_id()
        context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                            save_graphs=False)
    # define network
    net = xception(class_num=config.class_num)
    if args_opt.device_target == "Ascend":
        net.to_float(mstype.float16)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define dataset
    dataset = create_dataset(args_opt.train_data_dir, do_train=True, batch_size=config.batch_size,
                             device_num=group_size, rank=rank)
    step_size = dataset.get_dataset_size()

    # resume
    if args_opt.resume:
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net, ckpt)

    # get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode=config.lr_decay_mode,
                       global_step=config.finish_epoch * step_size))

    # define optimization and model
    if args_opt.device_target == "Ascend":
        opt = Momentum(net.trainable_params(), lr, config.momentum, config.weight_decay, config.loss_scale)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level='O3', keep_batchnorm_fp32=True)
    elif args_opt.device_target == "GPU":
        if args_opt.is_fp32:
            opt = Momentum(net.trainable_params(), lr, config.momentum, config.weight_decay)
            model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
        else:
            opt = Momentum(net.trainable_params(), lr, config.momentum, config.weight_decay, config.loss_scale)
            model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                          amp_level='O2', keep_batchnorm_fp32=True)

    # define callbacks
    cb = [TimeMonitor(), LossMonitor()]
    if config.save_checkpoint:
        if args_opt.device_target == "Ascend":
            save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        elif args_opt.device_target == "GPU":
            if args_opt.is_fp32:
                save_ckpt_path = os.path.join(config.save_checkpoint_path, 'fp32/' + 'model_' + str(rank))
            else:
                save_ckpt_path = os.path.join(config.save_checkpoint_path, 'fp16/' + 'model_' + str(rank))
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(f"Xception-rank{rank}", directory=save_ckpt_path, config=config_ck)

    # begin train
    print("begin train")
    if args_opt.is_distributed:
        if rank == 0:
            cb += [ckpt_cb]
        model.train(config.epoch_size - config.finish_epoch, dataset, callbacks=cb, dataset_sink_mode=True)
    else:
        cb += [ckpt_cb]
        model.train(config.epoch_size - config.finish_epoch, dataset, callbacks=cb, dataset_sink_mode=True)
    print("train success")


if __name__ == '__main__':
    run_train()
