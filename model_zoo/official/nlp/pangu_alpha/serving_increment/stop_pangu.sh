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

CURRUSER=$(whoami)

kill_serving_9()
{
  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "Send kill -9 msg to serving_server.py process"
  fi

  num=`ps -ef | grep start_distributed_worker.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'start_distributed_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "Send kill -9 msg to start_distributed_worker.py process"
  fi

  num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "Send kill -9 msg to serving_agent.py process"
  fi

  num=`ps -ef | grep 'flask/client.py' | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'flask/client.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "Send kill -9 msg to flask/client.py process"
  fi
}

kill_serving_15()
{
  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
    echo "Send kill -15 msg to serving_server.py process"
  fi

  num=`ps -ef | grep 'flask/client.py' | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'flask/client.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
    echo "Send kill -15 msg to flask/client.py process"
  fi

  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  count=0
  while [[ ${num} -ne 0 && ${count} -lt 5 ]]
  do
    sleep 1
    count=$(($count+1))
    num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  done

  num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  count=0
  while [[ ${num} -ne 0 && ${count} -lt 5 ]]
  do
    sleep 1
    count=$(($count+1))
    num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  done
}

kill_serving_15
kill_serving_9
