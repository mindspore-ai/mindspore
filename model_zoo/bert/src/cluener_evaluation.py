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
import tokenization
from sample_process import label_generation, process_one_example_p
from .evaluation_config import cfg
from .CRF import postprocess

vocab_file = "./vocab.txt"
tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)

def process(model, text, sequence_length):
    """
    process text.
    """
    data = [text]
    features = []
    res = []
    ids = []
    for i in data:
        feature = process_one_example_p(tokenizer_, i, max_seq_len=sequence_length)
        features.append(feature)
        input_ids, input_mask, token_type_id = feature
        input_ids = Tensor(np.array(input_ids), mstype.int32)
        input_mask = Tensor(np.array(input_mask), mstype.int32)
        token_type_id = Tensor(np.array(token_type_id), mstype.int32)
        if cfg.use_crf:
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
    res = label_generation(text, ids)
    return res

def submit(model, path, sequence_length):
    """
    submit task
    """
    data = []
    for line in open(path):
        if not line.strip():
            continue
        oneline = json.loads(line.strip())
        res = process(model, oneline["text"], sequence_length)
        print("text", oneline["text"])
        print("res:", res)
        data.append(json.dumps({"label": res}, ensure_ascii=False))
    open("ner_predict.json", "w").write("\n".join(data))
