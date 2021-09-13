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
"""cnnctc train"""


import numpy as np
import mindspore
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.callback import LossCallBack
from src.cnn_ctc import CNNCTC_Model, ctc_loss, WithLossCell, CNNCTCTrainOneStepWithLossScaleCell
from src.dataset import ST_MJ_Generator_batch_fixed_length, ST_MJ_Generator_batch_fixed_length_para
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper


set_seed(1)


context.set_context(mode=context.GRAPH_MODE, save_graphs=False, save_graphs_path=".")


def dataset_creator(run_distribute):
    if run_distribute:
        st_dataset = ST_MJ_Generator_batch_fixed_length_para()
    else:
        st_dataset = ST_MJ_Generator_batch_fixed_length()

    ds = GeneratorDataset(st_dataset,
                          ['img', 'label_indices', 'text', 'sequence_length'],
                          num_parallel_workers=8)

    return ds


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    target = config.device_target
    context.set_context(device_target=target)

    if target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)

        if config.run_distribute:
            init()
            context.set_auto_parallel_context(parallel_mode="data_parallel")

        ckpt_save_dir = config.SAVE_PATH
    else:
        # GPU target
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode="data_parallel",
                                              gradients_mean=False,
                                              gradient_fp32_sync=False)

            ckpt_save_dir = config.SAVE_PATH + "ckpt_" + str(get_rank()) + "/"
            print(ckpt_save_dir)
        else:
            ckpt_save_dir = config.SAVE_PATH + "ckpt_standalone/"

    ds = dataset_creator(config.run_distribute)

    net = CNNCTC_Model(config.NUM_CLASS, config.HIDDEN_SIZE, config.FINAL_FEATURE_WIDTH)
    net.set_train(True)

    if config.PRED_TRAINED:
        param_dict = load_checkpoint(config.PRED_TRAINED)
        load_param_into_net(net, param_dict)
        print('parameters loaded!')
    else:
        print('train from scratch...')

    criterion = ctc_loss()
    dataset_size = ds.get_dataset_size()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = mindspore.nn.RMSProp(params=net.trainable_params(),
                               centered=True,
                               learning_rate=lr,
                               momentum=config.MOMENTUM,
                               loss_scale=config.LOSS_SCALE)

    net = WithLossCell(net, criterion)

    if target == "Ascend":
        loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(
            config.LOSS_SCALE, False)
        net.set_train(True)
        model = Model(net, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O2")
    else:
        scaling_sens = Tensor(np.full((1), config.LOSS_SCALE), dtype=mstype.float32)
        net = CNNCTCTrainOneStepWithLossScaleCell(net, opt, scaling_sens)
        net.set_train(True)
        model = Model(net)

    callback = LossCallBack()
    config_ck = CheckpointConfig(save_checkpoint_steps=config.SAVE_CKPT_PER_N_STEP,
                                 keep_checkpoint_max=config.KEEP_CKPT_MAX_NUM)
    ckpoint_cb = ModelCheckpoint(prefix="CNNCTC", config=config_ck, directory=ckpt_save_dir)

    if config.run_distribute:
        if device_id == 0:
            model.train(config.TRAIN_EPOCHS,
                        ds,
                        callbacks=[callback, ckpoint_cb],
                        dataset_sink_mode=False)
        else:
            model.train(config.TRAIN_EPOCHS, ds, callbacks=[callback], dataset_sink_mode=False)
    else:
        model.train(config.TRAIN_EPOCHS,
                    ds,
                    callbacks=[callback, ckpoint_cb],
                    dataset_sink_mode=False)


if __name__ == '__main__':
    train()
