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
"""execute reader"""

from collections import defaultdict
import random
from time import time
import json
import numpy as np
from tqdm import tqdm

from transformers import AlbertTokenizer

from mindspore import Tensor, ops
from mindspore import dtype as mstype

from src.rerank_and_reader_data_generator import DataGenerator
from src.rerank_and_reader_utils import convert_to_tokens, make_wiki_id, DocDB
from src.reader import Reader


def read(args):
    """reader function"""
    db_file = args.wiki_db_path
    reader_feature_file = args.reader_feature_file
    reader_example_file = args.reader_example_file
    encoder_ck_file = args.reader_encoder_ck_path
    downstream_ck_file = args.reader_downstream_ck_path
    albert_model_path = args.albert_model_path
    reader_result_file = args.reader_result_file
    seed = args.seed
    sp_threshold = args.sp_threshold
    seq_len = args.seq_len
    batch_size = args.reader_batch_size
    para_limit = args.max_para_num
    sent_limit = args.max_sent_num

    random.seed(seed)
    np.random.seed(seed)

    t1 = time()

    doc_db = DocDB(db_file)

    generator = DataGenerator(feature_file_path=reader_feature_file,
                              example_file_path=reader_example_file,
                              batch_size=batch_size, seq_len=seq_len,
                              para_limit=para_limit, sent_limit=sent_limit,
                              task_type="reader")
    example_dict = generator.example_dict
    feature_dict = generator.feature_dict
    answer_dict = defaultdict(lambda: defaultdict(list))
    new_answer_dict = {}
    total_sp_dict = defaultdict(list)
    new_total_sp_dict = defaultdict(list)

    tokenizer = AlbertTokenizer.from_pretrained(albert_model_path)
    new_tokens = ['[q]', '[/q]', '<t>', '</t>', '[s]']
    tokenizer.add_tokens(new_tokens)

    reader = Reader(batch_size=batch_size,
                    encoder_ck_file=encoder_ck_file,
                    downstream_ck_file=downstream_ck_file)

    print("start reading ...")

    for _, batch in tqdm(enumerate(generator)):
        input_ids = Tensor(batch["context_idxs"], mstype.int32)
        attn_mask = Tensor(batch["context_mask"], mstype.int32)
        token_type_ids = Tensor(batch["segment_idxs"], mstype.int32)
        context_mask = Tensor(batch["context_mask"], mstype.float32)
        square_mask = Tensor(batch["square_mask"], mstype.float32)
        packing_mask = Tensor(batch["query_mapping"], mstype.float32)
        para_start_mapping = Tensor(batch["para_start_mapping"], mstype.float32)
        sent_end_mapping = Tensor(batch["sent_end_mapping"], mstype.float32)
        unique_ids = batch["unique_ids"]
        sent_names = batch["sent_names"]
        cache_mask = Tensor(np.tril(np.triu(np.ones((seq_len, seq_len)), 0), 30), mstype.float32)

        _, _, q_type, _, sent_logit, y1, y2 = reader(input_ids, attn_mask, token_type_ids,
                                                     context_mask, square_mask, packing_mask, cache_mask,
                                                     para_start_mapping, sent_end_mapping)

        type_prob = ops.Softmax()(q_type).asnumpy()

        answer_dict_ = convert_to_tokens(example_dict,
                                         feature_dict,
                                         batch['ids'],
                                         y1.asnumpy().tolist(),
                                         y2.asnumpy().tolist(),
                                         type_prob,
                                         tokenizer,
                                         sent_logit.asnumpy(),
                                         sent_names,
                                         unique_ids)
        for q_id in answer_dict_:
            answer_dict[q_id] = answer_dict_[q_id]

    for q_id in answer_dict:
        res = answer_dict[q_id]
        answer_text_ = res[0]
        sent_ = res[1]
        sent_names_ = res[2]
        new_answer_dict[q_id] = answer_text_

        predict_support_np = ops.Sigmoid()(Tensor(sent_, mstype.float32)).asnumpy()

        for j in range(predict_support_np.shape[0]):
            if j >= len(sent_names_):
                break
            if predict_support_np[j] > sp_threshold:
                total_sp_dict[q_id].append(sent_names_[j])

    for _id in total_sp_dict:
        _sent_names = total_sp_dict[_id]
        for para in _sent_names:
            title = make_wiki_id(para[0], 0)
            para_original_title = doc_db.get_doc_info(title)[-1]
            para[0] = para_original_title
            new_total_sp_dict[_id].append(para)

    prediction = {'answer': new_answer_dict,
                  'sp': new_total_sp_dict}

    with open(reader_result_file, 'w') as f:
        json.dump(prediction, f, indent=4)

    t2 = time()

    print(f"reader cost time: {t2-t1} s")
