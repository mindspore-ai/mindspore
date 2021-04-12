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
"""main file"""

import os
from time import time
from mindspore import context
from src.rerank_and_reader_utils import get_parse, cal_reranker_metrics, select_reader_dev_data
from src.reranker_eval import rerank
from src.reader_eval import read
from src.hotpot_evaluate_v1 import hotpotqa_eval
from src.build_reranker_data import get_rerank_data


def rerank_and_retriever_eval():
    """main function"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    parser = get_parse()
    args = parser.parse_args()
    args.dev_gold_path = os.path.join(args.data_path, args.dev_gold_file)
    args.wiki_db_path = os.path.join(args.data_path, args.wiki_db_file)
    args.albert_model_path = os.path.join(args.ckpt_path, args.albert_model)
    args.rerank_encoder_ck_path = os.path.join(args.ckpt_path, args.rerank_encoder_ck_file)
    args.rerank_downstream_ck_path = os.path.join(args.ckpt_path, args.rerank_downstream_ck_file)
    args.reader_encoder_ck_path = os.path.join(args.ckpt_path, args.reader_encoder_ck_file)
    args.reader_downstream_ck_path = os.path.join(args.ckpt_path, args.reader_downstream_ck_file)

    if args.get_reranker_data:
        get_rerank_data(args)

    if args.run_reranker:
        rerank(args)

    if args.cal_reranker_metrics:
        total_top1_pem, _, _ = \
            cal_reranker_metrics(dev_gold_file=args.dev_gold_path, rerank_result_file=args.rerank_result_file)

    if args.select_reader_data:
        select_reader_dev_data(args)

    if args.run_reader:
        read(args)

    if args.cal_reader_metrics:
        metrics = hotpotqa_eval(args.reader_result_file, args.dev_gold_path)

    if args.cal_reranker_metrics:
        print(f"total top1 pem: {total_top1_pem}")

    if args.cal_reader_metrics:
        for k in metrics:
            print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    t1 = time()
    rerank_and_retriever_eval()
    t2 = time()
    print(f"eval reranker and reader cost {(t2 - t1) / 3600} h")
