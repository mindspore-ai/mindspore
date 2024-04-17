#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

#!/bin/bash
pytest_bert()
{
  echo "###  pytest start ###"
  pytest -s -v test_bert_tdt_ge.py > test_bert_tdt_ge.log 2>&1 &
  if [ $? -ne 0 ]
  then
    cat test_bert_tdt_ge.log
    echo "pytest failed to start." && exit 1
  fi

  result=`ps -ef | grep test_bert_tdt_ge.py | grep -v grep | wc -l`
  count=0
  while [[ ${result} -ne 0 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`ps -ef | grep test_bert_tdt_ge.py | grep -v grep | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    ps aux | grep 'test_bert_tdt_ge.py' | grep -v grep | awk '{print $2}' | xargs kill -9
    sleep 1
    cat test_bert_tdt_ge.log
    echo "run test_bert_tdt_ge timeout!" && exit 1
  fi
  echo "### pytest end ###"
}

echo "### ge_bert start ###"
pytest_bert
echo "### end to ge_bert test ###"
exit 0
