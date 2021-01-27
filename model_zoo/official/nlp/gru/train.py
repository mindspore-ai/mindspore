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
"""train script"""
import os
import time
import argparse
import ast
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.nn.optim import Adam
from src.config import config
from src.seq2seq import Seq2Seq
from src.gru_for_train import GRUWithLossCell, GRUTrainOneStepWithLossScaleCell
from src.dataset import create_gru_dataset
from src.lr_schedule import dynamic_lr
set_seed(1)

parser = argparse.ArgumentParser(description="GRU training")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default: false.")
parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained file path.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
parser.add_argument('--ckpt_path', type=str, default='outputs/', help='Checkpoint save location. Default: outputs/')
parser.add_argument('--outputs_dir', type=str, default='./', help='Checkpoint save location. Default: outputs/')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id, save_graphs=False)

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))
time_stamp_init = False
time_stamp_first = 0
class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
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
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')

if __name__ == '__main__':
    if args.run_distribute:
        rank = args.rank_id
        device_num = args.device_num
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1
    mindrecord_file = args.dataset_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.batch_size,
                                 dataset_path=mindrecord_file, rank_size=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    print("dataset size is {}".format(dataset_size))
    network = Seq2Seq(config)
    network = GRUWithLossCell(network)
    lr = dynamic_lr(config, dataset_size)
    opt = Adam(network.trainable_params(), learning_rate=lr)
    scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale_value,
                                            scale_factor=config.scale_factor,
                                            scale_window=config.scale_window)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = GRUTrainOneStepWithLossScaleCell(network, opt, update_cell)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    #Save Checkpoint
    if config.save_checkpoint:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_epoch*dataset_size,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_'+str(args.rank_id)+'/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args.rank_id))
        cb += [ckpt_cb]
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(config.num_epochs, dataset, callbacks=cb, dataset_sink_mode=True)
