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
"""YoloV3 train."""
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
from mindspore.train.callback import CheckpointConfig
from mindspore import amp
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed

from src.yolo import YOLOV3DarkNet53, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov3_params
from src.util import keep_loss_fp32

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

set_seed(1)

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


def conver_training_shape(args):
    training_shape = [int(args.training_shape), int(args.training_shape)]
    return training_shape

def set_graph_kernel_context():
    if context.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                               "--enable_trans_op_optimize "
                                               "--disable_cluster_ops=ReduceMax,Reshape "
                                               "--enable_expand_ops=Conv2D")

def network_init(args):
    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target, save_graphs=False, device_id=devid)
    set_graph_kernel_context()

    profiler = None
    if args.need_profiler:
        from mindspore.profiler import Profiler
        profiling_dir = os.path.join("profiling",
                                     datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        profiler = Profiler(output_path=profiling_dir, is_detail=True, is_show_op_path=True)

    # init distributed
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
        else:
            init("nccl")
        args.rank = get_rank()
        args.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1
    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)
    return profiler


def parallel_init(args):
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)


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

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Train function."""
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoch
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.data_root = os.path.join(config.data_dir, 'train2014')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_train2014.json')

    profiler = network_init(config)

    loss_meter = AverageMeter('loss')
    parallel_init(config)

    network = YOLOV3DarkNet53(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov3_params(config, network)

    network = YoloWithLossCell(network)
    config.logger.info('finish get network')

    config.label_smooth = config.label_smooth
    config.label_smooth_factor = config.label_smooth_factor

    if config.training_shape:
        config.multi_scale = [conver_training_shape(config)]

    ds, data_size = create_yolo_dataset(image_dir=config.data_root, anno_path=config.annFile, is_training=True,
                                        batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                        device_num=config.group_size, rank=config.rank, config=config)
    config.logger.info('Finish loading dataset')

    config.steps_per_epoch = int(data_size / config.per_batch_size / config.group_size)

    if config.ckpt_interval <= 0:
        config.ckpt_interval = config.steps_per_epoch

    lr = get_lr(config)
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)
    is_gpu = context.get_context("device_target") == "GPU"
    if is_gpu:
        loss_scale_value = 1.0
        loss_scale = FixedLossScaleManager(loss_scale_value, drop_overflow_update=False)
        network = amp.build_train_network(network, optimizer=opt, loss_scale_manager=loss_scale,
                                          level="O2", keep_batchnorm_fp32=False)
        keep_loss_fp32(network)
    else:
        network = TrainingWrapper(network, opt, sens=config.loss_scale)
        network.set_train()

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        cb_params = InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        config.logger.info('iter[{}], shape{}'.format(i, input_shape[0]))

        images = Tensor.from_numpy(images)

        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_y_true_2 = Tensor.from_numpy(data['bbox3'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])
        batch_gt_box2 = Tensor.from_numpy(data['gt_box3'])

        loss = network(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                       batch_gt_box2)
        loss_meter.update(loss.asnumpy())

        if config.rank_save_ckpt_flag:
            # ckpt progress
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % config.log_interval == 0:
            time_used = time.time() - t_end
            epoch = int(i / config.steps_per_epoch)
            per_step_time = time_used/config.log_interval
            fps = config.per_batch_size * (i - old_progress) * config.group_size / time_used
            if config.rank == 0:
                config.logger.info(
                    'epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr:{},'
                    ' per_step_time:{}'.format(epoch, i, loss_meter, fps, lr[i], per_step_time))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % config.steps_per_epoch == 0 and config.rank_save_ckpt_flag:
            cb_params.cur_epoch_num += 1

        if config.need_profiler:
            if i == 10:
                profiler.analyse()
                break
    config.logger.info('==========end training===============')


if __name__ == "__main__":
    run_train()
