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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "sh process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE"
echo "for example: sh process_output.sh /path/newstest2014.tok.de /path/eval_output_file /path/vocab.bpe.32000"
echo "It is better to use absolute path."
echo "=============================================================================================================="

BASEDIR=$(dirname "$0")

ref_data=$1
eval_output=$2
vocab_file=$3

cat $eval_output \
  | python src/process_output.py --vocab_file $vocab_file \
  | sed 's/@@ //g' > ${eval_output}.processed

perl -ple 's/(\S)-(\S)/$1 #@#-#@# $2/g' < $ref_data | perl ${BASEDIR}/replace-quote.perl > ${ref_data}.forbleu
perl -ple 's/(\S)-(\S)/$1 #@#-#@# $2/g' < ${eval_output}.processed > ${eval_output}.forbleu