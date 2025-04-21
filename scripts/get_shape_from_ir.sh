#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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

set -e

# Usage : get_shape_from_ir.sh ir_file

cat "$1" | perl -p -e 's/\n/NEWLINE/' \
         | sed 's/NEWLINE      :/:/g' \
         | sed 's/Tensor NEWLINEshape//g' \
         | perl -p -e 's/NEWLINE/\n/g' \
         | perl -p -e 's/<Array\[([\d\w]+)\]x\[[\w ]+\](\[[\d, ]*\])>/\2/g' \
         | perl -p -e 's/<Tuple\[([\[\]\d\w\.\*]*)\]>/Tuple/g' \
         | perl -p -e 's/ \%(\d+)\(.*= /\1\t/g' \
         | perl -p -e 's/\(.*\)( \{.*\})*:/\t\1\t/g' \
         | tr -d '()' \
         | awk '/subgraph/{p=1;next}{if(p){print}}'\
         | awk '/return/{p=1;next}{if(!p){print}}' \
         | sed '/^$/d' \
         | awk -F'\t' '{print $1"\t"$2"\t"$4}'
