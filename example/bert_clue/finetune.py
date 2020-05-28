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

'''
Bert finetune script.
'''

import os
from utils import BertFinetuneCell, BertCLS, BertNER
from finetune_config import cfg, bert_net_cfg, tag_to_index
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecayDynamicLR, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net

class LossCallBack(Callback):
    '''
    Monitor the loss in training.
    If the loss is NAN or INF, terminate training.
    Note:
        If per_print_times is 0, do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    '''
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be in and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        with open("./loss.log", "a+") as f:
            f.write("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                                 str(cb_params.net_outputs)))
            f.write("\n")

def get_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    '''
    get dataset
    '''
    _ = distribute_file

    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask",
                                                                            "segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)

    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def test_train():
    '''
    finetune function
    pytest -s finetune.py::test_train
    '''
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    #BertCLSTrain for classification
    #BertNERTrain for sequence labeling
    if cfg.task == 'NER':
        if cfg.use_crf:
            netwithloss = BertNER(bert_net_cfg, True, num_labels=len(tag_to_index), use_crf=True,
                                  tag_to_index=tag_to_index, dropout_prob=0.1)
        else:
            netwithloss = BertNER(bert_net_cfg, True, num_labels=cfg.num_labels, dropout_prob=0.1)
    else:
        netwithloss = BertCLS(bert_net_cfg, True, num_labels=cfg.num_labels, dropout_prob=0.1)
    dataset = get_dataset(bert_net_cfg.batch_size, cfg.epoch_num)
    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    if cfg.optimizer == 'AdamWeightDecayDynamicLR':
        optimizer = AdamWeightDecayDynamicLR(netwithloss.trainable_params(),
                                             decay_steps=steps_per_epoch * cfg.epoch_num,
                                             learning_rate=cfg.AdamWeightDecayDynamicLR.learning_rate,
                                             end_learning_rate=cfg.AdamWeightDecayDynamicLR.end_learning_rate,
                                             power=cfg.AdamWeightDecayDynamicLR.power,
                                             warmup_steps=steps_per_epoch,
                                             weight_decay=cfg.AdamWeightDecayDynamicLR.weight_decay,
                                             eps=cfg.AdamWeightDecayDynamicLR.eps)
    elif cfg.optimizer == 'Lamb':
        optimizer = Lamb(netwithloss.trainable_params(), decay_steps=steps_per_epoch * cfg.epoch_num,
                         start_learning_rate=cfg.Lamb.start_learning_rate, end_learning_rate=cfg.Lamb.end_learning_rate,
                         power=cfg.Lamb.power, warmup_steps=steps_per_epoch, decay_filter=cfg.Lamb.decay_filter)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(netwithloss.trainable_params(), learning_rate=cfg.Momentum.learning_rate,
                             momentum=cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported.")
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.ckpt_prefix, directory=cfg.ckpt_dir, config=ckpt_config)
    param_dict = load_checkpoint(cfg.pre_training_ckpt)
    load_param_into_net(netwithloss, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertFinetuneCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    model.train(cfg.epoch_num, dataset, callbacks=[LossCallBack(), ckpoint_cb])
    D.release()

if __name__ == "__main__":
    test_train()
