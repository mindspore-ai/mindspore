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

"""Evaluation script for SQuAD task"""

import os
import collections
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src import tokenization
from src.evaluation_config import cfg, bert_net_cfg
from src.utils import BertSquad
from src.create_squad_data import read_squad_examples, convert_examples_to_features
from src.run_squad import write_predictions

def get_squad_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    """get SQuAD dataset from tfrecord"""
    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask",
                                                                            "segment_ids", "unique_ids"],
                            shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.repeat(repeat_count)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def test_eval():
    """Evaluation function for SQuAD task"""
    tokenizer = tokenization.FullTokenizer(vocab_file="./vocab.txt", do_lower_case=True)
    input_file = "dataset/v1.1/dev-v1.1.json"
    eval_examples = read_squad_examples(input_file, False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        output_fn=None,
        verbose_logging=False)

    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=device_id)
    dataset = get_squad_dataset(bert_net_cfg.batch_size, 1)
    net = BertSquad(bert_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator():
        input_data = []
        for i in columns_list:
            input_data.append(Tensor(data[i]))
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                               end_positions, unique_ids, is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(bert_net_cfg.batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    write_predictions(eval_examples, eval_features, output, 20, 30, True, "./predictions.json",
                      None, None, False, False)


if __name__ == "__main__":
    test_eval()
