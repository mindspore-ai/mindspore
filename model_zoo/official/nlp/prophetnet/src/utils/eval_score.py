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
"""Get score by given metric."""
from .ppl_score import ngram_ppl
from .rouge_score import rouge


def get_ppl_score(result):
    """
    Calculate Perplexity(PPL) score.

    Args:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "log_prob" and "length".

    Returns:
        Float, ppl score.
    """
    log_probs = []
    total_length = 0

    for sample in result:
        log_prob = sample['log_prob']
        length = sample['length']
        log_probs.extend(log_prob)
        total_length += length

        print(f" | log_prob:{log_prob}")
        print(f" | length:{length}")

    ppl = ngram_ppl(log_probs, total_length, log_softmax=True)
    print(f" | final PPL={ppl}.")
    return ppl


def get_rouge_score(result, vocab):
    """
    Calculate ROUGE score.

    Args:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "prediction" and "prediction_prob".
        Dictionary, dict instance.

    return:
        Str, rouge score.
    """

    predictions = []
    targets = []
    for sample in result:
        predictions.append(' '.join([vocab[t] for t in sample['prediction']]))
        targets.append(' '.join([vocab[t] for t in sample['target']]))
        print(f" | source: {' '.join([vocab[t] for t in sample['source']])}")
        print(f" | target: {targets[-1]}")

    return rouge(predictions, targets)


def get_score(result, vocab=None, metric='rouge'):
    """
    Get eval score.

    Args:
        List[Dict], prediction.
        Dictionary, dict instance.
        Str, metric function, default is rouge.

    Return:
        Str, Score.
    """
    score = None
    if metric == 'rouge':
        score = get_rouge_score(result, vocab)
    elif metric == 'ppl':
        score = get_ppl_score(result)
    else:
        print(f" |metric not in (rouge, ppl)")

    return score
