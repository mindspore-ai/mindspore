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
"""Face attribute train."""
import os
import time
import datetime
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import get_group_size, init, get_rank
from mindspore.nn import TrainOneStepCell
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, RunContext, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from src.FaceAttribute.resnet18 import get_resnet18
from src.FaceAttribute.loss_factory import get_loss
from src.dataset_train import data_generator
from src.lrsche_factory import warmup_step
from src.logging import get_logger, AverageMeter
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, _key):
        return self[_key]

    def __setattr__(self, _key, _value):
        self[_key] = _value


class BuildTrainNetwork(nn.Cell):
    '''Build train network.'''

    def __init__(self, my_network, my_criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = my_network
        self.criterion = my_criterion

    def construct(self, input_data, label):
        logit0, logit1, logit2 = self.network(input_data)
        loss0 = self.criterion(logit0, logit1, logit2, label)
        return loss0


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

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''run train.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                        device_id=get_device_id())
    mindspore.set_seed(1)

    # init distributed
    if config.world_size != 1:
        init()
        config.local_rank = get_rank()
        config.world_size = get_group_size()
        config.lr = config.lr * 4.
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        config.per_batch_size = 256
        parallel_mode = ParallelMode.STAND_ALONE

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=config.world_size)

    config.outputs_dir = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.local_rank)
    loss_meter = AverageMeter('loss')

    # dataloader
    config.logger.info('start create dataloader')
    de_dataloader, steps_per_epoch, num_classes = data_generator(config)
    config.steps_per_epoch = steps_per_epoch
    config.num_classes = num_classes
    config.logger.info('end create dataloader')
    config.logger.save_args(config)

    # backbone && loss && load pretrain model
    config.logger.important_info('start create network')
    create_network_start = time.time()
    network = get_resnet18(config)
    criterion = get_loss()
    if os.path.isfile(config.pretrained):
        param_dict = load_checkpoint(config.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        config.logger.info('load model %s success', config.pretrained)

    # optimizer and lr scheduler
    lr = warmup_step(config, gamma=0.1)
    opt = Momentum(params=network.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    train_net = BuildTrainNetwork(network, criterion)
    # mixed precision training
    criterion.add_flags_recursive(fp32=True)
    train_net = TrainOneStepCell(train_net, opt, sens=config.loss_scale)
    context.reset_auto_parallel_context()

    if config.local_rank == 0:
        ckpt_max_num = config.max_epoch
        train_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=train_config, directory=config.outputs_dir,
                                  prefix='{}'.format(config.local_rank))
        cb_params = InternalCallbackParam()
        cb_params.train_network = train_net
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 0
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1

    i = 0
    for _, (data, gt_classes) in enumerate(de_dataloader):

        data_tensor = Tensor(data, dtype=mstype.float32)
        gt_tensor = Tensor(gt_classes, dtype=mstype.int32)

        loss = train_net(data_tensor, gt_tensor)
        loss_meter.update(loss.asnumpy()[0])

        if config.local_rank == 0:
            cb_params.cur_step_num = i + 1
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % config.steps_per_epoch == 0 and config.local_rank == 0:
            cb_params.cur_epoch_num += 1

        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            config.logger.important_info(
                '{}, graph compile time={:.2f}s'.format(config.backbone, time_for_graph_compile))

        if i % config.log_interval == 0 and config.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / config.steps_per_epoch)
            fps = config.per_batch_size * (i - old_progress) * config.world_size / time_used
            config.logger.info('epoch[{}], iter[{}], {}, {:.2f} imgs/sec'.format(epoch, i, loss_meter, fps))

            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if i % config.steps_per_epoch == 0 and config.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / config.steps_per_epoch)
            fps = config.per_batch_size * config.world_size * config.steps_per_epoch / epoch_time_used
            config.logger.info('=================================================')
            config.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            config.logger.info('=================================================')
            t_epoch = time.time()

        i += 1

    config.logger.info('--------- trains out ---------')


if __name__ == "__main__":
    run_train()
