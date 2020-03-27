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

"""
NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) is the Chinese pretrained language model currently based on BERT developed by Huawei.
1. Prepare data
Following the data preparation as in BERT, run command as below to get dataset for training:
    python ./create_pretraining_data.py \
      --input_file=./sample_text.txt \
      --output_file=./examples.tfrecord \
      --vocab_file=./your/path/vocab.txt \
      --do_lower_case=True \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5
2. Pretrain
First, prepare the distributed training environment, then adjust configurations in config.py, finally run main.py.
"""

import os
import pytest
import numpy as np
from numpy import allclose
from config import bert_cfg as cfg
import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore._c_dataengine as deMap
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.model_zoo.Bert_NEZHA import BertConfig, BertNetworkWithLoss, BertTrainOneStepCell
from mindspore.nn.optim import Lamb
from mindspore import log as logger
_current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = [cfg.DATA_DIR]
SCHEMA_DIR = cfg.SCHEMA_DIR

def me_de_train_dataset(batch_size):
    """test me de train dataset"""
    # apply repeat operations
    repeat_count = cfg.epoch_size
    ds = de.StorageDataset(DATA_DIR, SCHEMA_DIR, columns_list=["input_ids", "input_mask", "segment_ids",
                                                               "next_sentence_labels", "masked_lm_positions",
                                                               "masked_lm_ids", "masked_lm_weights"])
    type_cast_op = deMap.TypeCastOp("int32")
    ds = ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    ds = ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    ds = ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)
    return ds


def weight_variable(shape):
    """weight variable"""
    np.random.seed(1)
    ones = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
    return Tensor(ones)


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs.asnumpy()[0])
        logger.info("epoch: {}, outputs are {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))

def test_bert_tdt():
    """test bert tdt"""
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="Ascend")
    context.set_context(enable_task_sink=True)
    context.set_context(enable_loop_sink=True)
    context.set_context(enable_mem_reuse=True)
    parallel_callback = ModelCallback()
    ds = me_de_train_dataset(cfg.bert_config.batch_size)
    config = cfg.bert_config
    netwithloss = BertNetworkWithLoss(config, True)
    optimizer = Lamb(netwithloss.trainable_params(), decay_steps=cfg.decay_steps, start_learning_rate=cfg.start_learning_rate,
                     end_learning_rate=cfg.end_learning_rate, power=cfg.power, warmup_steps=cfg.num_warmup_steps, decay_filter=lambda x: False)
    netwithgrads = BertTrainOneStepCell(netwithloss, optimizer=optimizer)
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.checkpoint_prefix, config=config_ck)
    model.train(ds.get_repeat_count(), ds, callbacks=[parallel_callback, ckpoint_cb], dataset_sink_mode=False)

if __name__ == '__main__':
    test_bert_tdt()
