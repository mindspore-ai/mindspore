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

'''bert clue evaluation'''

import json
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from src import tokenization
from src.sample_process import label_generation, process_one_example_p
from src.CRF import postprocess
from src.model_utils.config import bert_net_cfg
from src.score import get_result

def process(model=None, text="", tokenizer_=None, use_crf="", tag_to_index=None, vocab=""):
    """
    process text.
    """
    data = [text]
    features = []
    res = []
    ids = []
    for i in data:
        feature = process_one_example_p(tokenizer_, vocab, i, max_seq_len=bert_net_cfg.seq_length)
        features.append(feature)
        input_ids, input_mask, token_type_id = feature
        input_ids = Tensor(np.array(input_ids), mstype.int32)
        input_mask = Tensor(np.array(input_mask), mstype.int32)
        token_type_id = Tensor(np.array(token_type_id), mstype.int32)
        if use_crf.lower() == "true":
            backpointers, best_tag_id = model.predict(input_ids, input_mask, token_type_id, Tensor(1))
            best_path = postprocess(backpointers, best_tag_id)
            logits = []
            for ele in best_path:
                logits.extend(ele)
            ids = logits
        else:
            logits = model.predict(input_ids, input_mask, token_type_id, Tensor(1))
            ids = logits.asnumpy()
            ids = np.argmax(ids, axis=-1)
            ids = list(ids)
    res = label_generation(text=text, probs=ids, tag_to_index=tag_to_index)
    return res

def submit(model=None, path="", vocab_file="", use_crf="", label_file="", tag_to_index=None):
    """
    submit task
    """
    tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
    data = []
    for line in open(path):
        if not line.strip():
            continue
        oneline = json.loads(line.strip())
        res = process(model=model, text=oneline["text"], tokenizer_=tokenizer_,
                      use_crf=use_crf, tag_to_index=tag_to_index, vocab=vocab_file)
        data.append(json.dumps({"label": res}, ensure_ascii=False))
    open("ner_predict.json", "w").write("\n".join(data))
    labels = []
    with open(label_file) as f:
        for label in f:
            labels.append(label.strip())
    get_result(labels, "ner_predict.json", path)
