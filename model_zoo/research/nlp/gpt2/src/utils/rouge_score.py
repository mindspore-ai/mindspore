"""Calculate ROUGE score."""
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

from typing import List
from rouge import Rouge


def get_rouge_score(hypothesis: List[str], target: List[str]):
    """
    Calculate ROUGE score.

    Args:
        hypothesis (List[str]): Inference result.
        target (List[str]): Reference.
    """
    if not hypothesis or not target:
        raise ValueError(f"`hypothesis` and `target` can not be None.")
    _rouge = Rouge()
    print("hypothesis:", hypothesis)
    print("target:", target)
    scores = _rouge.get_scores(hypothesis, target, avg=True)
    print(" | ROUGE Score:")
    print(f" | RG-1(F): {scores['rouge-1']['f'] * 100:8.2f}")
    print(f" | RG-2(F): {scores['rouge-2']['f'] * 100:8.2f}")
    print(f" | RG-L(F): {scores['rouge-l']['f'] * 100:8.2f}")
    return scores
