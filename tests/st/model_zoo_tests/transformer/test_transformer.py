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
"""Transformer testing script."""

import time
import os
import pytest
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import Callback
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import context
from easydict import EasyDict as edict
from tests.models.official.nlp.transformer.src.transformer_model import TransformerConfig
from tests.models.official.nlp.transformer.src.transformer_for_train import TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell
from tests.models.official.nlp.transformer.src.lr_schedule import create_dynamic_lr
from tests.st.model_zoo_tests import utils


DATA_DIR = ["/home/workspace/mindspore_dataset/transformer/test-mindrecord"]

cfg = edict({
    'transformer_network': 'large',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,
    'optimizer': 'Adam',
    'optimizer_adam_beta2': 0.997,
    'lr_schedule': edict({
        'learning_rate': 2.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
})


def get_config(version='base', batch_size=1):
    """get config"""
    if version == 'large':
        transformer_cfg = TransformerConfig(
            batch_size=96,
            seq_length=128,
            vocab_size=36560,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="relu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=128,
            initializer_range=0.02,
            label_smoothing=0.1,
            dtype=mstype.float32,
            compute_type=mstype.float16)
    elif version == 'base':
        transformer_cfg = TransformerConfig(
            batch_size=96,
            seq_length=128,
            vocab_size=36560,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act="relu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=128,
            initializer_range=0.02,
            label_smoothing=0.1,
            dtype=mstype.float32,
            compute_type=mstype.float16)
    else:
        transformer_cfg = TransformerConfig(batch_size=batch_size)
    return transformer_cfg


def load_test_data(batch_size=1, data_file=None):
    """Load test dataset."""
    data_set = ds.MindDataset(data_file,
                              columns_list=["source_eos_ids", "source_eos_mask",
                                            "target_sos_ids", "target_sos_mask",
                                            "target_eos_ids", "target_eos_mask"],
                              shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="source_eos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="source_eos_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_sos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_sos_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_eos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_eos_mask")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []
        self.overflow_list = []
        self.lossscale_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs[0].asnumpy()[0])
        self.overflow_list.append(cb_params.net_outputs[1].asnumpy())
        self.lossscale_list.append(cb_params.net_outputs[2].asnumpy())
        print("epoch: {}, outputs are: {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))


class TimeMonitor(Callback):
    """Time Monitor."""

    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_mseconds_list = []
        self.per_step_mseconds_list = []

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds_list.append(epoch_mseconds)
        self.per_step_mseconds_list.append(epoch_mseconds / self.data_size)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transformer():
    """
    Transformer training.
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(reserve_class_name_in_scope=False, enable_auto_mixed_precision=False)
    version = os.getenv('VERSION', 'large')
    batch_size = 96
    epoch_size = 3
    config = get_config(version=version, batch_size=batch_size)
    dataset = load_test_data(batch_size=config.batch_size, data_file=DATA_DIR)

    netwithloss = TransformerNetworkWithLoss(config, True)

    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size() * epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  hidden_size=config.hidden_size), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    callback = ModelCallback()

    scale_manager = DynamicLossScaleManager(init_loss_scale=4194304,
                                            scale_factor=cfg.scale_factor,
                                            scale_window=3)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                            scale_update_cell=update_cell)

    netwithgrads.set_train(True)
    time_monitor_callback = TimeMonitor(dataset.get_dataset_size())
    model = Model(netwithgrads)
    model.train(epoch_size, dataset, callbacks=[time_monitor_callback, callback], dataset_sink_mode=False)

    # assertion occurs while the loss value, overflow state or loss_scale value is wrong
    loss_value = np.array(callback.loss_list)
    assert np.allclose(loss_value[0], 11.241601, 0, 0.000005)

    expect_loss_value = [11.241606, 11.243232, 11.217459, 11.204157, 11.213804,
                         11.215373, 11.190564, 11.150393, 11.191823, 11.160045]

    print("loss value: {}".format(loss_value))
    assert np.allclose(loss_value[0:10], expect_loss_value, 0, 0.0005)

    overflow = np.array(callback.overflow_list)
    expect_overflow = [False, False, False, True, False, False, False, True, False, False]
    print("overflow: {}".format(overflow))
    assert (overflow[0:10] == expect_overflow).all()

    loss_scale = np.array(callback.lossscale_list)
    expect_loss_scale = [4194304.0, 4194304.0, 8388608.0, 4194304.0, 4194304.0,
                         4194304.0, 8388608.0, 4194304.0, 4194304.0, 4194304.0]
    print("loss scale: {}".format(loss_scale))
    assert np.allclose(loss_scale[0:10], expect_loss_scale, 0, 0)

    epoch_mseconds = np.array(time_monitor_callback.epoch_mseconds_list)[2]
    expect_epoch_mseconds = 2400
    print("epoch mseconds: {}".format(epoch_mseconds))
    assert epoch_mseconds <= expect_epoch_mseconds + 100

    per_step_mseconds = np.array(time_monitor_callback.per_step_mseconds_list)[2]
    expect_per_step_mseconds = 240
    print("per step mseconds: {}".format(per_step_mseconds))
    assert per_step_mseconds <= expect_per_step_mseconds + 10


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transformer_export_mindir():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/nlp".format(cur_path)
    model_name = "transformer"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    export_file = "transformer80_bs_0"
    ckpt_path = os.path.join(utils.ckpt_root, "transformer/transformer_trained.ckpt")
    print("ckpt_path:", ckpt_path)
    old_list = ["model_file: './transformer/transformer_trained.ckpt'"]
    new_list = ["model_file: '{}'".format(ckpt_path)]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "default_config_large.yaml"))
    old_list = ["context.set_context(device_id=get_device_id())"]
    new_list = ["context.set_context()"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "export.py"))
    exec_export_shell = "cd transformer; python -u export.py --file_name={}" \
                        " --file_format=MINDIR --config_path=./default_config_large.yaml".format(export_file)
    os.system(exec_export_shell)
    assert os.path.exists(os.path.join(cur_model_path, "{}.mindir".format(export_file)))

if __name__ == '__main__':
    test_transformer()
