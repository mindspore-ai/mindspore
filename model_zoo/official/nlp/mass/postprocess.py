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
"""Evaluation api."""
import os
import pickle
import numpy as np

from src.model_utils.config import config
from src.utils import Dictionary
from src.utils import get_score

def read_from_file():
    '''
     calculate accuraty.
    '''
    predictions = []
    probs = []
    source_sentences = []
    target_sentences = []
    file_num = len(os.listdir(config.source_id_folder))
    for i in range(file_num):
        f_name = "gigaword_bs_" + str(config.batch_size) + "_" + str(i)
        source_ids = np.fromfile(os.path.join(config.source_id_folder, f_name + ".bin"), np.int32)
        source_ids = source_ids.reshape(1, config.max_decode_length)
        target_ids = np.fromfile(os.path.join(config.target_id_folder, f_name + ".bin"), np.int32)
        target_ids = target_ids.reshape(1, config.max_decode_length)
        predicted_ids = np.fromfile(os.path.join(config.result_dir, f_name + "_0.bin"), np.int32)
        predicted_ids = predicted_ids.reshape(1, config.max_decode_length + 1)
        entire_probs = np.fromfile(os.path.join(config.result_dir, f_name + "_1.bin"), np.float32)
        entire_probs = entire_probs.reshape(1, config.beam_width, config.max_decode_length + 1)

        source_sentences.append(source_ids)
        target_sentences.append(target_ids)
        predictions.append(predicted_ids)
        probs.append(entire_probs)

    output = []
    for inputs, ref, batch_out, batch_probs in zip(source_sentences,
                                                   target_sentences,
                                                   predictions,
                                                   probs):
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]

            example = {
                "source": inputs[i].tolist(),
                "target": ref[i].tolist(),
                "prediction": batch_out[i].tolist(),
                "prediction_prob": batch_probs[i].tolist()
            }
            output.append(example)

    return output


if __name__ == '__main__':
    result = read_from_file()
    vocab = Dictionary.load_from_persisted_dict(config.vocab)

    with open(config.output, "wb") as f:
        pickle.dump(result, f, 1)

    # get score by given metric
    score = get_score(result, vocab, metric=config.metric)
    print(score)
