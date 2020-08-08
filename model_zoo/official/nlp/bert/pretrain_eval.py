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
Bert evaluation script.
"""

import os
from src import BertModel, GetMaskedLMOutput
from src.evaluation_config import cfg, bert_net_cfg
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.nn as nn
from mindspore.nn.metrics import Metric
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

class myMetric(Metric):
    '''
    Self-defined Metric as a callback.
    '''
    def __init__(self):
        super(myMetric, self).__init__()
        self.clear()

    def clear(self):
        self.total_num = 0
        self.acc_num = 0

    def update(self, *inputs):
        total_num = self._convert_data(inputs[0])
        acc_num = self._convert_data(inputs[1])
        self.total_num = total_num
        self.acc_num = acc_num

    def eval(self):
        return self.acc_num/self.total_num


class GetLogProbs(nn.Cell):
    '''
    Get MaskedLM prediction scores
    '''
    def __init__(self, config):
        super(GetLogProbs, self).__init__()
        self.bert = BertModel(config, False)
        self.cls1 = GetMaskedLMOutput(config)

    def construct(self, input_ids, input_mask, token_type_id, masked_pos):
        sequence_output, _, embedding_table = self.bert(input_ids, token_type_id, input_mask)
        prediction_scores = self.cls1(sequence_output, embedding_table, masked_pos)
        return prediction_scores


class BertPretrainEva(nn.Cell):
    '''
    Evaluate MaskedLM prediction scores
    '''
    def __init__(self, config):
        super(BertPretrainEva, self).__init__()
        self.bert = GetLogProbs(config)
        self.argmax = P.Argmax(axis=-1, output_type=mstype.int32)
        self.equal = P.Equal()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.total = Parameter(Tensor([0], mstype.float32), name='total')
        self.acc = Parameter(Tensor([0], mstype.float32), name='acc')
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()


    def construct(self, input_ids, input_mask, token_type_id, masked_pos, masked_ids, masked_weights, nsp_label):
        bs, _ = self.shape(input_ids)
        probs = self.bert(input_ids, input_mask, token_type_id, masked_pos)
        index = self.argmax(probs)
        index = self.reshape(index, (bs, -1))
        eval_acc = self.equal(index, masked_ids)
        eval_acc1 = self.cast(eval_acc, mstype.float32)
        real_acc = eval_acc1 * masked_weights
        acc = self.sum(real_acc)
        total = self.sum(masked_weights)
        self.total += total
        self.acc += acc
        return acc, self.total, self.acc


def get_enwiki_512_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    '''
    Get enwiki seq_length=512 dataset
    '''
    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask", "segment_ids",
                                                                            "masked_lm_positions", "masked_lm_ids",
                                                                            "masked_lm_weights",
                                                                            "next_sentence_labels"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    ds = ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    ds = ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    ds = ds.repeat(repeat_count)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def bert_predict():
    '''
    Predict function
    '''
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    dataset = get_enwiki_512_dataset(bert_net_cfg.batch_size, 1)
    net_for_pretraining = BertPretrainEva(bert_net_cfg)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset, net_for_pretraining


def MLM_eval():
    '''
    Evaluate function
    '''
    _, dataset, net_for_pretraining = bert_predict()
    net = Model(net_for_pretraining, eval_network=net_for_pretraining, eval_indexes=[0, 1, 2],
                metrics={'name': myMetric()})
    res = net.eval(dataset, dataset_sink_mode=False)
    print("==============================================================")
    for _, v in res.items():
        print("Accuracy is: ")
        print(v)
    print("==============================================================")


if __name__ == "__main__":
    MLM_eval()
