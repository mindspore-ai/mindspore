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
"""Calculate ROUGE score."""
from typing import List
from rouge import Rouge

H_PATH = "summaries.txt"
R_PATH = "references.txt"


def rouge(hypothesis: List[str], target: List[str]):
    """
    Calculate ROUGE score.

    Args:
        hypothesis (List[str]): Inference result.
        target (List[str]): Reference.
    """

    def cut(s):
        idx = s.find("</s>")
        if idx != -1:
            s = s[:idx]
        return s

    if not hypothesis or not target:
        raise ValueError(f"`hypothesis` and `target` can not be None.")

    edited_hyp = []
    edited_ref = []
    for h, r in zip(hypothesis, target):
        h = cut(h).replace("<s>", "").strip()
        r = cut(r).replace("<s>", "").strip()
        edited_hyp.append(h + "\n")
        edited_ref.append(r + "\n")

    _rouge = Rouge()
    scores = _rouge.get_scores(edited_hyp, edited_ref, avg=True)
    print(" | ROUGE Score:")
    print(f" | RG-1(F): {scores['rouge-1']['f'] * 100:8.2f}")
    print(f" | RG-2(F): {scores['rouge-2']['f'] * 100:8.2f}")
    print(f" | RG-L(F): {scores['rouge-l']['f'] * 100:8.2f}")

    with open(H_PATH, "w") as f:
        f.writelines(edited_hyp)

    with open(R_PATH, "w") as f:
        f.writelines(edited_ref)
