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
Retriever Evaluation.

"""

import time
import json

import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net

from src.onehop import OneHopBert
from src.twohop import TwoHopBert
from src.process_data import DataGen
from src.onehop_bert import ModelOneHop
from src.twohop_bert import ModelTwoHop
from src.config import ThinkRetrieverConfig
from src.utils import read_query, split_queries, get_new_title, get_raw_title, save_json


def eval_output(out_2, last_out, path_raw, gold_path, val, true_count):
    """evaluation output"""
    y_pred_raw = out_2.asnumpy()
    last_out_raw = last_out.asnumpy()
    path = []
    y_pred = []
    last_out_list = []
    topk_titles = []
    index_list_raw = np.argsort(y_pred_raw)
    for index_r in index_list_raw[::-1]:
        tag = 1
        for raw_path in path:
            if path_raw[index_r][0] in raw_path and path_raw[index_r][1] in raw_path:
                tag = 0
                break
        if tag:
            path.append(path_raw[index_r])
            y_pred.append(y_pred_raw[index_r])
            last_out_list.append(last_out_raw[index_r])
    index_list = np.argsort(y_pred)
    for path_index in index_list:
        if gold_path[0] in path[path_index] and gold_path[1] in path[path_index]:
            true_count += 1
            break
    for path_index in index_list[-8:][::-1]:
        topk_titles.append(list(path[path_index]))
    for path_index in index_list[-8:]:
        if gold_path[0] in path[path_index] and gold_path[1] in path[path_index]:
            val += 1
            break
    return val, true_count, topk_titles


def evaluation():
    """evaluation"""
    print('********************** loading corpus ********************** ')
    s_lc = time.time()
    data_generator = DataGen(config)
    queries = read_query(config)
    print("loading corpus time (h):", (time.time() - s_lc) / 3600)
    print('********************** loading model ********************** ')
    s_lm = time.time()

    model_onehop_bert = ModelOneHop()
    param_dict = load_checkpoint(config.onehop_bert_path)
    load_param_into_net(model_onehop_bert, param_dict)
    model_twohop_bert = ModelTwoHop()
    param_dict2 = load_checkpoint(config.twohop_bert_path)
    load_param_into_net(model_twohop_bert, param_dict2)
    onehop = OneHopBert(config, model_onehop_bert)
    twohop = TwoHopBert(config, model_twohop_bert)

    print("loading model time (h):", (time.time() - s_lm) / 3600)
    print('********************** evaluation ********************** ')
    s_tr = time.time()

    f_dev = open(config.dev_path, 'rb')
    dev_data = json.load(f_dev)
    q_gold = {}
    q_2id = {}
    for onedata in dev_data:
        if onedata["question"] not in q_gold:
            q_gold[onedata["question"]] = [get_new_title(get_raw_title(item)) for item in onedata['path']]
            q_2id[onedata["question"]] = onedata['_id']
    val, true_count, count, step = 0, 0, 0, 0
    batch_queries = split_queries(config, queries)[:-1]
    output_path = []
    for _, batch in enumerate(batch_queries):
        print("###step###: ", step)
        query = batch[0]
        temp_dict = {}
        temp_dict['q_id'] = q_2id[query]
        temp_dict['question'] = query
        gold_path = q_gold[query]
        input_ids_1, token_type_ids_1, input_mask_1 = data_generator.convert_onehop_to_features(batch)
        start = 0
        TOTAL = len(input_ids_1)
        split_chunk = 8
        while start < TOTAL:
            end = min(start + split_chunk - 1, TOTAL - 1)
            chunk_len = end - start + 1
            input_ids_1_ = input_ids_1[start:start + chunk_len]
            input_ids_1_ = Tensor(input_ids_1_, mstype.int32)
            token_type_ids_1_ = token_type_ids_1[start:start + chunk_len]
            token_type_ids_1_ = Tensor(token_type_ids_1_, mstype.int32)
            input_mask_1_ = input_mask_1[start:start + chunk_len]
            input_mask_1_ = Tensor(input_mask_1_, mstype.int32)
            cls_out = onehop(input_ids_1_, token_type_ids_1_, input_mask_1_)
            if start == 0:
                out = cls_out
            else:
                out = P.Concat(0)((out, cls_out))
            start = end + 1
        out = P.Squeeze(1)(out)
        onehop_prob, onehop_index = P.TopK(sorted=True)(out, config.topk)
        onehop_prob = P.Softmax()(onehop_prob)
        sample, path_raw, last_out = data_generator.get_samples(query, onehop_index, onehop_prob)
        input_ids_2, token_type_ids_2, input_mask_2 = data_generator.convert_twohop_to_features(sample)
        start_2 = 0
        TOTAL_2 = len(input_ids_2)
        split_chunk = 8
        while start_2 < TOTAL_2:
            end_2 = min(start_2 + split_chunk - 1, TOTAL_2 - 1)
            chunk_len = end_2 - start_2 + 1
            input_ids_2_ = input_ids_2[start_2:start_2 + chunk_len]
            input_ids_2_ = Tensor(input_ids_2_, mstype.int32)
            token_type_ids_2_ = token_type_ids_2[start_2:start_2 + chunk_len]
            token_type_ids_2_ = Tensor(token_type_ids_2_, mstype.int32)
            input_mask_2_ = input_mask_2[start_2:start_2 + chunk_len]
            input_mask_2_ = Tensor(input_mask_2_, mstype.int32)
            cls_out = twohop(input_ids_2_, token_type_ids_2_, input_mask_2_)
            if start_2 == 0:
                out_2 = cls_out
            else:
                out_2 = P.Concat(0)((out_2, cls_out))
            start_2 = end_2 + 1
        out_2 = P.Softmax()(out_2)
        last_out = Tensor(last_out, mstype.float32)
        out_2 = P.Mul()(out_2, last_out)
        val, true_count, topk_titles = eval_output(out_2, last_out, path_raw, gold_path, val, true_count)
        temp_dict['topk_titles'] = topk_titles
        output_path.append(temp_dict)
        count += 1
        print("val:", val)
        print("count:", count)
        print("true count:", true_count)
        if count:
            print("PEM:", val / count)
        if true_count:
            print("true top8 PEM:", val / true_count)
        step += 1
    save_json(output_path, config.save_path, config.save_name)
    print("evaluation time (h):", (time.time() - s_tr) / 3600)


if __name__ == "__main__":
    config = ThinkRetrieverConfig()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend',
                        device_id=config.device_id,
                        save_graphs=False)
    evaluation()
