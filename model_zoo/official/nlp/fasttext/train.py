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
"""FastText for train"""
import os
import time
import argparse
from mindspore import context
from mindspore.nn.optim import Adam
from mindspore.common import set_seed
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.communication import management as MultiAscend
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.load_dataset import load_dataset
from src.lr_schedule import polynomial_decay_scheduler
from src.fasttext_train import FastTextTrainOneStepCell, FastTextNetWithLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='FastText input data file path.')
parser.add_argument('--data_name', type=str, required=True, default='ag', help='dataset name. eg. ag, dbpedia')
args = parser.parse_args()

if args.data_name == "ag":
    from src.config import config_ag as config
elif args.data_name == 'dbpedia':
    from src.config import config_db as config
elif args.data_name == 'yelp_p':
    from  src.config import config_yelpp as config

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))
set_seed(5)
time_stamp_init = False
time_stamp_first = 0
rank_id = os.getenv('DEVICE_ID')
context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend")

class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1, rank_ids=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_ids
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs.asnumpy())))
            f.write('\n')


def _build_training_pipeline(pre_dataset):
    """
    Build training pipeline

    Args:
        pre_dataset: preprocessed dataset
    """
    net_with_loss = FastTextNetWithLoss(config.vocab_size, config.embedding_dims, config.num_class)
    net_with_loss.init_parameters_data()
    if config.pretrain_ckpt_dir:
        parameter_dict = load_checkpoint(config.pretrain_ckpt_dir)
        load_param_into_net(net_with_loss, parameter_dict)
    if pre_dataset is None:
        raise ValueError("pre-process dataset must be provided")

    #get learning rate
    update_steps = config.epoch * pre_dataset.get_dataset_size()
    decay_steps = pre_dataset.get_dataset_size()
    rank_size = os.getenv("RANK_SIZE")
    if isinstance(rank_size, int):
        raise ValueError("RANK_SIZE must be integer")
    if rank_size is not None and int(rank_size) > 1:
        base_lr = config.lr
    else:
        base_lr = config.lr / 10
    print("+++++++++++Total update steps ", update_steps)
    lr = Tensor(polynomial_decay_scheduler(lr=base_lr,
                                           min_lr=config.min_lr,
                                           decay_steps=decay_steps,
                                           total_update_num=update_steps,
                                           warmup_steps=config.warmup_steps,
                                           power=config.poly_lr_scheduler_power), dtype=mstype.float32)
    optimizer = Adam(net_with_loss.trainable_params(), lr, beta1=0.9, beta2=0.999)

    net_with_grads = FastTextTrainOneStepCell(net_with_loss, optimizer=optimizer)
    net_with_grads.set_train(True)
    model = Model(net_with_grads)
    loss_monitor = LossCallBack(rank_ids=rank_id)
    dataset_size = pre_dataset.get_dataset_size()
    time_monitor = TimeMonitor(data_size=dataset_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=decay_steps * config.epoch,
                                   keep_checkpoint_max=config.keep_ckpt_max)
    callbacks = [time_monitor, loss_monitor]
    if rank_size is None or int(rank_size) == 1:
        ckpt_callback = ModelCheckpoint(prefix='fasttext',
                                        directory=os.path.join('./', 'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                        config=ckpt_config)
        callbacks.append(ckpt_callback)
    if rank_size is not None and int(rank_size) > 1 and MultiAscend.get_rank() % 8 == 0:
        ckpt_callback = ModelCheckpoint(prefix='fasttext',
                                        directory=os.path.join('./', 'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                        config=ckpt_config)
        callbacks.append(ckpt_callback)
    print("Prepare to Training....")
    epoch_size = pre_dataset.get_repeat_count()
    print("Epoch size ", epoch_size)
    if os.getenv("RANK_SIZE") is not None and int(os.getenv("RANK_SIZE")) > 1:
        print(f" | Rank {MultiAscend.get_rank()} Call model train.")
    model.train(epoch=config.epoch, train_dataset=pre_dataset, callbacks=callbacks, dataset_sink_mode=False)


def train_single(input_file_path):
    """
    Train model on single device
    Args:
        input_file_path: preprocessed dataset path
    """
    print("Staring training on single device.")
    preprocessed_data = load_dataset(dataset_path=input_file_path,
                                     batch_size=config.batch_size,
                                     epoch_count=config.epoch_count,
                                     bucket=config.buckets)
    _build_training_pipeline(preprocessed_data)


def set_parallel_env():
    context.reset_auto_parallel_context()
    MultiAscend.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      device_num=MultiAscend.get_group_size(),
                                      gradients_mean=True)
def train_paralle(input_file_path):
    """
    Train model on multi device
    Args:
        input_file_path: preprocessed dataset path
    """
    set_parallel_env()
    print("Starting traning on mutiple devices. |~ _ ~| |~ _ ~| |~ _ ~| |~ _ ~|")
    preprocessed_data = load_dataset(dataset_path=input_file_path,
                                     batch_size=config.batch_size,
                                     epoch_count=config.epoch_count,
                                     rank_size=MultiAscend.get_group_size(),
                                     rank_id=MultiAscend.get_rank(),
                                     bucket=config.buckets,
                                     shuffle=False)
    _build_training_pipeline(preprocessed_data)

if __name__ == "__main__":
    _rank_size = os.getenv("RANK_SIZE")
    if _rank_size is not None and int(_rank_size) > 1:
        train_paralle(args.data_path)
    else:
        train_single(args.data_path)
