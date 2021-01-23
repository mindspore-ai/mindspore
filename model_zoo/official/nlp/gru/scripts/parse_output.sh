#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE"
echo "for example: sh parse_output.sh target.txt output.txt vocab.en"
echo "It is better to use absolute path."
echo "=============================================================================================================="
ref_data=$1
eval_output=$2
vocab_file=$3

cat $ref_data \
  | python ../src/parse_output.py --vocab_file $vocab_file \
  | sed 's/@@ //g' > ${ref_data}.forbleu

cat $eval_output \
  | python ../src/parse_output.py --vocab_file $vocab_file \
  | sed 's/@@ //g' > ${eval_output}.forbleu