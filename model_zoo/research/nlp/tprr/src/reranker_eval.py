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
"""execute reranker"""

import json
import random
from collections import defaultdict
from time import time
import numpy as np
from tqdm import tqdm

from mindspore import Tensor, ops
from mindspore import dtype as mstype

from src.rerank_and_reader_data_generator import DataGenerator
from src.reranker import Reranker


def rerank(args):
    """rerank function"""
    rerank_feature_file = args.rerank_feature_file
    rerank_result_file = args.rerank_result_file
    encoder_ck_file = args.rerank_encoder_ck_path
    downstream_ck_file = args.rerank_downstream_ck_path
    seed = args.seed
    seq_len = args.seq_len
    batch_size = args.rerank_batch_size

    random.seed(seed)
    np.random.seed(seed)

    t1 = time()

    generator = DataGenerator(feature_file_path=rerank_feature_file,
                              example_file_path=None,
                              batch_size=batch_size, seq_len=seq_len,
                              task_type="reranker")
    gather_dict = defaultdict(lambda: defaultdict(list))

    reranker = Reranker(batch_size=batch_size,
                        encoder_ck_file=encoder_ck_file,
                        downstream_ck_file=downstream_ck_file)

    print("start re-ranking ...")

    for _, batch in tqdm(enumerate(generator)):
        input_ids = Tensor(batch["context_idxs"], mstype.int32)
        attn_mask = Tensor(batch["context_mask"], mstype.int32)
        token_type_ids = Tensor(batch["segment_idxs"], mstype.int32)

        no_answer = reranker(input_ids, attn_mask, token_type_ids)

        no_answer_prob = ops.Softmax()(no_answer).asnumpy()
        no_answer_prob = no_answer_prob[:, 0]

        for i in range(len(batch['ids'])):
            qas_id = batch['ids'][i]
            gather_dict[qas_id][no_answer_prob[i]].append(batch['unique_ids'][i])
            gather_dict[qas_id][no_answer_prob[i]].append(batch['path'][i])

    rerank_result = {}
    for qas_id in tqdm(gather_dict, desc="get top1 path from re-rank result"):
        all_paths = gather_dict[qas_id]
        all_paths = sorted(all_paths.items(), key=lambda item: item[0])
        assert qas_id not in rerank_result
        rerank_result[qas_id] = all_paths[0][1]

    with open(rerank_result_file, 'w') as f:
        json.dump(rerank_result, f)

    t2 = time()

    print(f"re-rank cost time: {t2-t1} s")
