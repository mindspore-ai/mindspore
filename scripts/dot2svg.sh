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
# set -x
# Usage: ./scripts/dot2svg.sh [search_path]
if [[ -z "$1" ]]
  then
    DIR="."
  else
    DIR="$1"
fi

for f in "${DIR}"/*.dot
do
dot -Tsvg -o "${DIR}/$(basename "${f}").svg" "${DIR}/$(basename "${f}")"
done
