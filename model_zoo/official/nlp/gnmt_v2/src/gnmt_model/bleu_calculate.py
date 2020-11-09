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
"""Calculate the blue scores"""
import subprocess
import numpy as np

from src.dataset.tokenizer import Tokenizer


def load_result_data(result_npy_addr):
    # load the numpy to list.
    result = np.load(result_npy_addr, allow_pickle=True)
    return result


def get_bleu_data(tokenizer: Tokenizer, result_npy_addr):
    """
    Detokenizer the prediction.

    Args:
        tokenizer (Tokenizer): tokenizer operations.
        result_npy_addr (string): Path to the predict file.

    Returns:
        List, the predict text context.
    """

    result = load_result_data(result_npy_addr)
    prediction_list = []
    for _, info in enumerate(result):
        # prediction detokenize
        prediction = info["prediction"]
        prediction_str = tokenizer.detokenize(prediction)
        prediction_list.append(prediction_str)

    return prediction_list


def calculate_sacrebleu(predict_path, target_path):
    """
    Calculate the BLEU scores.

    Args:
        predict_path (string): Path to the predict file.
        target_path (string): Path to the target file.

    Returns:
        Float32, bleu scores.
    """

    sacrebleu_params = '--score-only -lc --tokenize intl'
    sacrebleu = subprocess.run([f'sacrebleu --input {predict_path} \
                                {target_path} {sacrebleu_params}'],
                               stdout=subprocess.PIPE, shell=True)
    bleu_scores = round(float(sacrebleu.stdout.strip()), 2)
    return bleu_scores


def bleu_calculate(tokenizer, result_npy_addr, target_addr=None):
    """
    Calculate the BLEU scores.

    Args:
        tokenizer (Tokenizer): tokenizer operations.
        result_npy_addr (string): Path to the predict file.
        target_addr (string): Path to the target file.

    Returns:
        Float32, bleu scores.
    """

    prediction = get_bleu_data(tokenizer, result_npy_addr)
    print("predict:\n", prediction)

    eval_path = './predict.txt'
    with open(eval_path, 'w') as eval_file:
        lines = [line + '\n' for line in prediction]
        eval_file.writelines(lines)
    reference_path = target_addr
    bleu_scores = calculate_sacrebleu(eval_path, reference_path)
    return bleu_scores
