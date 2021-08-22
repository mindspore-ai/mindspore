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
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.nn.optim import Adam
from mindspore import log as logger

from src.seq2seq import Seq2Seq
from src.gru_for_train import GRUWithLossCell, GRUTrainOneStepWithLossScaleCell, GRUTrainOneStepCell
from src.dataset import create_gru_dataset
from src.lr_schedule import dynamic_lr

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_rank_id, get_device_id, get_device_num

set_seed(1)

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
            if context.get_context("device_target") == "Ascend":
                f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                    time_stamp_current - time_stamp_first,
                    cb_params.cur_epoch_num,
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs[0].asnumpy()),
                    str(cb_params.net_outputs[1].asnumpy()),
                    str(cb_params.net_outputs[2].asnumpy())))
            else:
                f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                    time_stamp_current - time_stamp_first,
                    cb_params.cur_epoch_num,
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs.asnumpy())))
            f.write('\n')


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

    config.outputs_dir = os.path.join(config.output_path, config.outputs_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """run train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        device_id=get_device_id(), save_graphs=False)
    if config.device_target == "GPU":
        if config.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            config.compute_type = mstype.float32

    device_num = get_device_num()
    if config.run_distribute:
        if config.device_target == "Ascend":
            rank = get_rank_id()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        elif config.device_target == "GPU":
            rank = get_rank()
            init("nccl")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            raise ValueError(config.device_target)
    else:
        rank = 0
        device_num = 1

    mindrecord_file = config.dataset_path
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
    if config.device_target == "Ascend":
        netwithgrads = GRUTrainOneStepWithLossScaleCell(network, opt, update_cell)
    else:
        netwithgrads = GRUTrainOneStepCell(network, opt)
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    # Save Checkpoint
    if config.save_checkpoint:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_epoch * dataset_size,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(rank))
        cb += [ckpt_cb]
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(config.num_epochs, dataset, callbacks=cb, dataset_sink_mode=True)

if __name__ == '__main__':
    run_train()
