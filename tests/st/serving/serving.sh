#!/bin/bash

export GLOG_v=1
export DEVICE_ID=1

MINDSPORE_INSTALL_PATH=$1
CURRPATH=$(cd $(dirname $0); pwd)
CURRUSER=$(whoami)
PROJECT_PATH=${CURRPATH}/../../../
ENV_DEVICE_ID=$DEVICE_ID
echo "MINDSPORE_INSTALL_PATH:"  ${MINDSPORE_INSTALL_PATH}
echo "CURRPATH:"  ${CURRPATH}
echo "CURRUSER:"  ${CURRUSER}
echo "PROJECT_PATH:"  ${PROJECT_PATH}
echo "ENV_DEVICE_ID:" ${ENV_DEVICE_ID}

MODEL_PATH=${CURRPATH}/model
export LD_LIBRARY_PATH=${MINDSPORE_INSTALL_PATH}/lib:/usr/local/python/python375/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=${MINDSPORE_INSTALL_PATH}/:${PYTHONPATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}
echo "-------------show MINDSPORE_INSTALL_PATH----------------"
ls -l ${MINDSPORE_INSTALL_PATH}
echo "------------------show /usr/lib64/----------------------"
ls -l /usr/local/python/python375/lib/

clean_pid()
{
  ps aux | grep 'ms_serving' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  if [ $? -ne 0 ]
  then
    echo "clean pip failed"
  fi
  sleep 6
}

prepare_model()
{
  echo "### begin to generate mode for serving test ###"
  python3 generate_model.py &> generate_model_serving.log
  echo "### end to generate mode for serving test ###"
  result=`ls -l | grep -E '*mindir' | grep -v ".log" | wc -l`
  if [ ${result} -ne 2 ]
  then
    cat generate_model_serving.log
    echo "### generate model for serving test failed ###" && exit 1
    clean_pid
  fi
  rm -rf model
  mkdir model
  mv *.mindir ${CURRPATH}/model
  cp ${MINDSPORE_INSTALL_PATH}/ms_serving ./
}

start_service()
{
  ${CURRPATH}/ms_serving --port=$1 --model_path=${MODEL_PATH} --model_name=$2 --device_id=$3 > $2_service.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "$2 faile to start."
  fi

  result=`grep -E 'MS Serving listening on 0.0.0.0:5500|MS Serving listening on 0.0.0.0:5501' $2_service.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'MS Serving listening on 0.0.0.0:5500|MS Serving listening on 0.0.0.0:5501' $2_service.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_pid
    cat $2_service.log
    echo "start serving service failed!" && exit 1
  fi
  echo "### start serving service end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  CLIENT_DEVICE_ID=$((${ENV_DEVICE_ID}+1))
  export DEVICE_ID=${CLIENT_DEVICE_ID}
  local test_client_name=$1
  echo "### $1 client start ###"
  python3 -m pytest -v -s client_example.py::${test_client_name} > ${test_client_name}_client.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_pid
    cat ${test_client_name}_client.log
    echo "client $1 faile to start." && exit 1
  fi
  echo "### $1 client end ###"
}

test_add_model()
{
  start_service 5500 add.mindir ${ENV_DEVICE_ID}
  pytest_serving test_add
  clean_pid
}

test_bert_model()
{
  start_service 5500 bert.mindir ${ENV_DEVICE_ID}
  pytest_serving test_bert
  clean_pid
}

echo "-----serving start-----"
rm -rf ms_serving *.log *.mindir *.dat ${CURRPATH}/model ${CURRPATH}/kernel_meta
prepare_model
test_add_model
test_bert_model
