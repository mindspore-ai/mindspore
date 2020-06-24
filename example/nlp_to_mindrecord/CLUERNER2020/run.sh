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

rm -f output/train.mindrecord*
rm -f output/dev.mindrecord*

if [ ! -d "../../../third_party/to_mindrecord/CLUERNER2020" ]; then
    echo "The patch base dir ../../../third_party/to_mindrecord/CLUERNER2020 is not exist."
    exit 1
fi

if [ ! -f "../../../third_party/patch/to_mindrecord/CLUERNER2020/data_processor_seq.patch" ]; then
    echo "The patch file ../../../third_party/patch/to_mindrecord/CLUERNER2020/data_processor_seq.patch is not exist."
    exit 1
fi

# patch for data_processor_seq.py
patch -p0 -d ../../../third_party/to_mindrecord/CLUERNER2020/ -o data_processor_seq_patched.py < ../../../third_party/patch/to_mindrecord/CLUERNER2020/data_processor_seq.patch
if [ $? -ne 0 ]; then
    echo "Patch ../../../third_party/to_mindrecord/CLUERNER2020/data_processor_seq.py failed"
    exit 1
fi

# use patched script
python ../../../third_party/to_mindrecord/CLUERNER2020/data_processor_seq_patched.py \
--vocab_file=../../../third_party/to_mindrecord/CLUERNER2020/vocab.txt \
--label2id_file=../../../third_party/to_mindrecord/CLUERNER2020/label2id.json
