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
"""YoloV4 train."""
import os
import time
import datetime

from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
import mindspore as ms
from mindspore import amp
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed

from src.yolo import YOLOV4TinyCspDarkNet53, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov4_params
from src.util import keep_loss_fp32
from src.eval_utils import apply_eval, EvalCallBack

from model_utils.config import config

set_seed(1)


def set_default():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.t_max:
        config.t_max = config.max_epoch

    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.data_root = os.path.join(config.data_dir, config.train_img_dir)
    config.annFile = os.path.join(config.data_dir, config.train_json_file)

    config.data_val_root = os.path.join(config.data_dir, config.val_img_dir)
    config.ann_val_file = os.path.join(config.data_dir, config.val_json_file)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=config.device_target, save_graphs=False, device_id=device_id)

    # init distributed
    if config.is_distributed:
        if config.device_target == "Ascend":
            init()
        else:
            init("nccl")
        config.rank = get_rank()
        config.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.ckpt_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)


def convert_training_shape(args_training_shape):
    training_shape = [int(args_training_shape), int(args_training_shape)]
    return training_shape


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network_, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network_
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss_ = self.criterion(output, label)
        return loss_


def get_network(net, cfg, learning_rate):
    opt = Momentum(params=get_param_groups(net),
                   learning_rate=Tensor(learning_rate),
                   momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay,
                   loss_scale=cfg.loss_scale)
    is_gpu = context.get_context("device_target") == "GPU"
    if is_gpu:
        loss_scale_value = 1.0
        loss_scale = FixedLossScaleManager(loss_scale_value, drop_overflow_update=False)
        net = amp.build_train_network(net, optimizer=opt, loss_scale_manager=loss_scale,
                                      level="O2", keep_batchnorm_fp32=False)
        keep_loss_fp32(net)
    else:
        net = TrainingWrapper(net, opt)
        net.set_train()

    return net


def run_train():
    set_default()
    loss_meter = AverageMeter('loss')
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
    network = YOLOV4TinyCspDarkNet53()
    if config.run_eval:
        network_eval = network
    # default is kaiming-normal
    default_recurisive_init(network)
    config.logger.info('resume_yolov4_tiny: %s', config.resume_yolov4)
    load_yolov4_params(config, network)
    network = YoloWithLossCell(network)
    config.logger.info('finish get network')
    if config.training_shape:
        config.multi_scale = [convert_training_shape(config.training_shape)]
    ds, data_size = create_yolo_dataset(image_dir=config.data_root, anno_path=config.annFile, is_training=True,
                                        batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                        device_num=config.group_size, rank=config.rank, config=config)
    config.logger.info('Finish loading dataset')
    config.steps_per_epoch = int(data_size / config.per_batch_size / config.group_size)
    if config.ckpt_interval <= 0:
        config.ckpt_interval = config.steps_per_epoch
    lr = get_lr(config)
    network = get_network(network, config, lr)
    network.set_train(True)
    if config.rank_save_ckpt_flag or config.run_eval:
        cb_params = _InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = 10
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        ckpt_cb.begin(run_context)

    if config.run_eval:
        data_val_root = config.data_val_root
        ann_val_file = config.ann_val_file
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        input_val_shape = Tensor(tuple(config.test_img_shape), ms.float32)
        # init detection engine
        eval_dataset, eval_data_size = create_yolo_dataset(data_val_root, ann_val_file, is_training=False,
                                                           batch_size=config.per_batch_size, max_epoch=1, device_num=1,
                                                           rank=0, shuffle=False, config=config)
        eval_param_dict = {"net": network_eval, "dataset": eval_dataset, "data_size": eval_data_size,
                           "anno_json": ann_val_file, "input_shape": input_val_shape, "args": config}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_ckpt_path, besk_ckpt_name="best_map.ckpt",
                               metrics_name="mAP")

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        config.logger.info('iter[%d], shape%d', i, input_shape[0])

        images = Tensor.from_numpy(images)
        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])

        input_shape = Tensor(tuple(input_shape[::-1]), ms.float32)
        loss = network(images, batch_y_true_0, batch_y_true_1, batch_gt_box0, batch_gt_box1, input_shape)
        loss_meter.update(loss.asnumpy())

        # ckpt progress
        if config.rank_save_ckpt_flag or config.run_eval:
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % config.log_interval == 0:
            time_used = time.time() - t_end
            epoch = int(i / config.steps_per_epoch)
            fps = config.per_batch_size * (i - old_progress) * config.group_size / time_used
            if config.rank == 0:
                config.logger.info(
                    'epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr:{}'.format(epoch, i, loss_meter, fps, lr[i]))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % config.steps_per_epoch == 0 and (config.run_eval or config.rank_save_ckpt_flag):
            if config.run_eval:
                eval_cb.epoch_end(run_context)
                network.set_train()
            cb_params.cur_epoch_num += 1

    config.logger.info('==========end training===============')


if __name__ == "__main__":
    run_train()
