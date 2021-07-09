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
"""Convert ids to tokens."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tokenization
from model_utils.config import config

# Explicitly set the encoding
sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=True)
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=True)

def main():

    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=config.vocab_file)

    for line in sys.stdin:
        token_ids = [int(x) for x in line.strip().split()]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        sent = " ".join(tokens)
        sent = sent.split("<s>")[-1]
        sent = sent.split("</s>")[0]
        print(sent.strip())

if __name__ == "__main__":
    main()
