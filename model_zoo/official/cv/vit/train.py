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
"""training script"""

import os
import time
import socket
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
from mindspore.profiler.profiling import Profiler
from mindspore.train.serialization import load_checkpoint
import mindspore.dataset as ds

from src.vit import get_network
from src.dataset import get_dataset
from src.cross_entropy import get_loss
from src.optimizer import get_optimizer
from src.lr_generator import get_lr
from src.eval_engine import get_eval_engine
from src.callback import StateMonitor
from src.logging import get_logger

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


try:
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')

    device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    local_rank = int(os.getenv('RANK_ID'))    # local_rank
    device_num = int(os.getenv('RANK_SIZE'))  # world_size
    print("distribute training")
except TypeError:
    device_id = 0   # 0 ~ 7
    local_rank = 0    # local_rank
    device_num = 1  # world_size
    print("standalone training")

def add_static_args(args):
    """add_static_args"""
    args.weight_decay = float(args.weight_decay)

    args.eval_engine = 'imagenet'
    args.split_point = 0.4
    args.poly_power = 2
    args.aux_factor = 0.4
    args.seed = 1
    args.auto_tune = 0

    if args.eval_offset < 0:
        args.eval_offset = args.max_epoch % args.eval_interval

    args.device_id = device_id
    args.local_rank = local_rank
    args.device_num = device_num
    args.dataset_name = 'imagenet'

    return args

def modelarts_pre_process():
    '''modelarts pre process function.'''
    start_t = time.time()

    val_file = os.path.join(config.data_path, 'val/imagenet_val.tar')
    train_file = os.path.join(config.data_path, 'train/imagenet_train.tar')
    tar_files = [val_file, train_file]

    print('tar_files:{}'.format(tar_files))
    for tar_file in tar_files:
        if os.path.exists(tar_file):
            t1 = time.time()
            tar_dir = os.path.dirname(tar_file)
            print('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            os.system('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            t2 = time.time()
            print('uncompress, time used={:.2f}s'.format(t2 - t1))
            os.system('cd {}; rm -rf {}'.format(tar_dir, tar_file))
        else:
            print('file no exists:', tar_file)

    end_t = time.time()
    print('tar cost time {:.2f} sec'.format(end_t-start_t))


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_net():
    """train_net"""
    args = add_static_args(config)
    np.random.seed(args.seed)
    args.logger = get_logger(args.save_checkpoint_path, rank=local_rank)

    context.set_context(device_id=device_id,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False)

    if args.auto_tune:
        context.set_context(auto_tune_mode='GA')
    elif args.device_num == 1:
        pass
    else:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if args.open_profiler:
        profiler = Profiler(output_path="data_{}".format(local_rank))

    # init the distribute env
    if not args.auto_tune and args.device_num > 1:
        init()

    # network
    net = get_network(backbone_name=args.backbone, args=args)

    # set grad allreduce split point
    parameters = [param for param in net.trainable_params()]
    parameter_len = len(parameters)
    if args.split_point > 0:
        print("split_point={}".format(args.split_point))
        split_parameter_index = [int(args.split_point*parameter_len),]
        parameter_indices = 1
        for i in range(parameter_len):
            if i in split_parameter_index:
                parameter_indices += 1
            parameters[i].comm_fusion = parameter_indices
    else:
        print("warning!!!, no split point")

    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, net, strict_load=False)

    # loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    loss = get_loss(loss_name=args.loss_name, args=args)

    # train dataset
    epoch_size = args.max_epoch
    dataset = get_dataset(dataset_name=args.dataset_name,
                          do_train=True,
                          dataset_path=args.dataset_path,
                          args=args)
    ds.config.set_seed(args.seed)
    step_size = dataset.get_dataset_size()
    args.steps_per_epoch = step_size

    # evaluation dataset
    eval_dataset = get_dataset(dataset_name=args.dataset_name,
                               do_train=False,
                               dataset_path=args.eval_path,
                               args=args)

    # evaluation engine
    if args.auto_tune or args.open_profiler or eval_dataset is None or args.device_num == 1:
        args.eval_engine = ''
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # loss scale
    loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    # learning rate
    lr_array = get_lr(global_step=0, lr_init=args.lr_init, lr_end=args.lr_min, lr_max=args.lr_max,
                      warmup_epochs=args.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                      lr_decay_mode=args.lr_decay_mode, poly_power=args.poly_power)
    lr = Tensor(lr_array)

    # optimizer, group_params used in grad freeze
    opt, _ = get_optimizer(optimizer_name=args.opt,
                           network=net,
                           lrs=lr,
                           args=args)

    # model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  metrics=eval_engine.metric, eval_network=eval_engine.eval_network,
                  loss_scale_manager=loss_scale, amp_level="O3")
    eval_engine.set_model(model)
    args.logger.save_args(args)

    t0 = time.time()
    # equal to model._init(dataset, sink_size=step_size)
    eval_engine.compile(sink_size=step_size)

    t1 = time.time()
    args.logger.info('compile time used={:.2f}s'.format(t1 - t0))

    # callbacks
    state_cb = StateMonitor(data_size=step_size,
                            tot_batch_size=args.batch_size * device_num,
                            lrs=lr_array,
                            eval_interval=args.eval_interval,
                            eval_offset=args.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)

    cb = [state_cb,]
    if args.save_checkpoint and local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_epochs*step_size,
                                     keep_checkpoint_max=args.keep_checkpoint_max,
                                     async_save=True)
        ckpt_cb = ModelCheckpoint(prefix=args.backbone, directory=args.save_checkpoint_path, config=config_ck)
        cb += [ckpt_cb]

    t0 = time.time()
    model.train(epoch_size, dataset, callbacks=cb, sink_size=step_size)
    t1 = time.time()
    args.logger.info('training time used={:.2f}s'.format(t1 - t0))
    last_metric = 'last_metric[{}]'.format(state_cb.best_acc)
    args.logger.info(last_metric)

    is_cloud = args.enable_modelarts
    if is_cloud:
        ip = os.getenv("BATCH_TASK_CURRENT_HOST_IP")
    else:
        ip = socket.gethostbyname(socket.gethostname())
    args.logger.info('ip[{}], mean_fps[{:.2f}]'.format(ip, state_cb.mean_fps))

    if args.open_profiler:
        profiler.analyse()

if __name__ == '__main__':
    train_net()
