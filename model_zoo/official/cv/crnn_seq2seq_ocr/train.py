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
"""
CRNN-Seq2Seq-OCR train.

"""
import datetime
import time
import os

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_ocr_train_dataset
from src.logger import get_logger
from src.attention_ocr import AttentionOCR, AttentionOCRWithLossCell, TrainingWrapper

from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

set_seed(1)
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

    if config.modelarts_dataset_unzip_name:
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

    config.ckpt_path = os.path.join(config.output_path, str(get_rank_id()), config.ckpt_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    """Train function."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    if config.is_distributed:
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1

    # Logger
    config.outputs_dir = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, rank)
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # DATASET
    prefix = "fsns.mindrecord"
    if config.enable_modelarts:
        mindrecord_file = os.path.join(config.data_path, prefix + "0")
    else:
        mindrecord_file = os.path.join(config.train_data_dir, prefix + "0")
    dataset = create_ocr_train_dataset(mindrecord_file,
                                       config.batch_size,
                                       rank_size=device_num,
                                       rank_id=rank)
    config.steps_per_epoch = dataset.get_dataset_size()
    config.logger.info('Finish loading dataset')

    if not config.ckpt_interval:
        config.ckpt_interval = config.steps_per_epoch
    config.logger.save_args(config)

    network = AttentionOCR(config.batch_size,
                           int(config.img_width / 4),
                           config.encoder_hidden_size,
                           config.decoder_hidden_size,
                           config.decoder_output_size,
                           config.max_length,
                           config.dropout_p)

    if config.pre_checkpoint_path:
        config.pre_checkpoint_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.pre_checkpoint_path
                                                  )
        param_dict = load_checkpoint(config.pre_checkpoint_path)
        load_param_into_net(network, param_dict)

    network = AttentionOCRWithLossCell(network, config.max_length)

    lr = Tensor(config.lr, mstype.float32)
    opt = nn.Adam(network.trainable_params(), lr, beta1=config.adam_beta1, beta2=config.adam_beta2,
                  loss_scale=config.loss_scale)

    network = TrainingWrapper(network, opt, sens=config.loss_scale)

    config.logger.info('Finished get network')

    callback = [TimeMonitor(data_size=1), LossMonitor()]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.outputs_dir, 'checkpoints' + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix="crnn_seq2seq_ocr")
        callback.append(ckpt_cb)

    model = Model(network)
    model.train(config.num_epochs, dataset, callbacks=callback)

    config.logger.info('==========Training Done===============')


if __name__ == "__main__":
    train()
