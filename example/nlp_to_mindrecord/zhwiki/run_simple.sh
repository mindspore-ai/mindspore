#!/bin/bash
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

rm -f output/simple.mindrecord*

if [ ! -d "../../../third_party/to_mindrecord/zhwiki" ]; then
    echo "The patch base dir ../../../third_party/to_mindrecord/zhwiki is not exist."
    exit 1
fi

if [ ! -f "../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch" ]; then
    echo "The patch file ../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch is not exist."
    exit 1
fi

# patch for create_pretraining_data.py
patch -p0 -d ../../../third_party/to_mindrecord/zhwiki/ -o create_pretraining_data_patched.py < ../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch
if [ $? -ne 0 ]; then
    echo "Patch ../../../third_party/to_mindrecord/zhwiki/create_pretraining_data.py failed"
    exit 1
fi

# using patched script to generate mindrecord
python ../../../third_party/to_mindrecord/zhwiki/create_pretraining_data_patched.py \
--input_file=../../../third_party/to_mindrecord/zhwiki/sample_text.txt \
--output_file=output/simple.mindrecord \
--partition_number=4 \
--vocab_file=../../../third_party/to_mindrecord/zhwiki/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=10    # user defined
