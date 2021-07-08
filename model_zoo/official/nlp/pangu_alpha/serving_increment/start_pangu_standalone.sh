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

unset http_proxy
unset https_proxy
export GLOG_v=1

start_serving_server()
{
  echo "### start serving server, see serving_server.log for detail ###"
  python3 pangu_standalone/serving_server.py > serving_server.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "serving server failed to start."
  fi

  result=`grep -E 'Master server start success' serving_server.log | wc -l`
  count=0
  while [[ ${result} -eq 0 && ${count} -lt 100 ]]
  do
    sleep 1

    num=`ps -ef | grep 'serving_server.py' | grep -v grep | wc -l`
    if [ $num -eq 0 ]
    then
      echo "start serving server failed, see log serving_server.log for more detail" && exit 1
    fi

    count=$(($count+1))
    result=`grep -E 'Master server start success' serving_server.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    echo "start serving server failed, see log serving_server.log for more detail" && exit 1
  fi
  echo "### start serving server end ###"
}

start_flask()
{
  echo "### start flask server, see flask.log for detail ###"
  python3 flask/client.py > flask.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "flask server failed to start."
  fi

  result=`grep -E 'Press CTRL\+C to quit' flask.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 10 ]]
  do
    sleep 1

    num=`ps -ef | grep 'flask/client.py' | grep -v grep | wc -l`
    if [ $num -eq 0 ]
    then
      bash stop_pangu.sh
      echo "start flask server failed, see log flask.log for more detail" && exit 1
    fi

    count=$(($count+1))
    result=`grep -E 'Press CTRL\+C to quit' flask.log | wc -l`
  done

  if [ ${count} -eq 10 ]
  then
    bash stop_pangu.sh
    echo "start flask server failed, see log flask.log for more detail" && exit 1
  fi
  echo "### start flask server end ###"
  cat flask.log
}

wait_serving_ready()
{
  echo "### waiting serving server ready, see and serving_logs/*.log for detail ###"
  result=`grep -E 'gRPC server start success' serving_server.log | wc -l`
  count=0
  while [[ ${result} -eq 0 && ${count} -lt 1800 ]]
  do
    sleep 1

    num=`ps -ef | grep 'serving_server.py' | grep -v grep | wc -l`
    if [ $num -eq 0 ]
    then
      bash stop_pangu.sh
      echo "waiting serving server ready failed, see log serving_server.log and serving_logs/*.log for more detail" && exit 1
    fi

    count=$(($count+1))
    result=`grep -E 'gRPC server start success' serving_server.log | wc -l`
  done

  if [ ${count} -eq 1800 ]
  then
    bash stop_pangu.sh
    echo "waiting serving server ready failed, see log serving_server.log and serving_logs/*.log for more detail" && exit 1
  fi
  echo "### waiting serving server ready end ###"
}

bash stop_pangu.sh
rm -rf serving_server.log flask.log serving_logs
start_serving_server
wait_serving_ready
start_flask
