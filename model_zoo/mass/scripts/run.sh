#!/usr/bin/env bash
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

export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

options=`getopt -u -o ht:n:i:j:c:o:v: -l help,task:,device_num:,device_id:,hccl_json:,config:,output:,vocab -- "$@"`
eval set -- "$options"
echo $options

echo_help()
{
  echo "Usage:"
  echo "bash train.sh [-h] [-t t|i] [-n N] [-i N] [-j FILE] [-c FILE] [-o FILE] [-v FILE]"
  echo "options:"
  echo "        -h --help                show usage"
  echo "        -t --task                select task, 't' for training and 'i' for inference"
  echo "        -n --device_num          training with N devices"
  echo "        -i --device_id           training with device i"
  echo "        -j --hccl_json           set the rank table file"
  echo "        -c --config              set the configuration file"
  echo "        -o --output              set the output file of inference"
  echo "        -v --vocab               set the vocabulary"
}

set_hccl_json()
{
  while [ -n "$1" ]
  do
    if [[ "$1" == "-j" || "$1"  == "--hccl_json" ]]
    then
      export MINDSPORE_HCCL_CONFIG_PATH=$2 #/data/wsc/hccl_2p_01.json
      export RANK_TABLE_FILE=$2 #/data/wsc/hccl_2p_01.json
      break
    fi
    shift
  done
}
set_device_id()
{
  while [ -n "$1" ]
  do
    if [[ "$1" == "-i" || "$1" == "--device_id" ]]
    then
      if [[ $2 -ge 0 && $2 -le 7 ]]
      then
        export DEVICE_ID=$2
      fi
      break
    fi
    shift
  done
}

while [ -n "$1" ]
do
  case "$1" in
  -h|--help)
      echo_help
      shift
      ;;
  -t|--task)
    echo "task:"
    if [ "$2" == "t" ]
    then
      task=train
    elif [ "$2" == "i" ]
    then
      task=infer
    fi
    shift 2
    ;;
  -n|--device_num)
    echo "device_num"
    if [ $2 -eq 1 ]
    then
      set_device_id $options
    elif [ $2 -gt 1 ]
    then
        export HCCL_FLAG=1
        export DEPLOY_MODE=0

        export RANK_SIZE=$2
        set_hccl_json $options
    fi
    shift 2
    ;;
  -i|--device_id)
    echo "set device id"
    export DEVICE_ID=$2
    shift 2
    ;;
  -c|--config)
    echo "config";
    configurations=$2
    shift 2
    ;;
  -o|--output)
    echo "output";
    output=$2
    shift 2
    ;;
  -v|--vocab)
    echo "vocab";
    vocab=$2
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *)
    shift
    ;;
esac
done

for((i=0; i < $RANK_SIZE; i++))
do
  if [ $RANK_SIZE -gt 1 ]
  then
    echo $RANK_SIZE
    export RANK_ID=$i
    export DEVICE_ID=$[i]
  fi
  echo "Working on device $i"

  file_path=$(cd "$(dirname $0)" || exit; pwd)
  cd $file_path || exit
  cd ../ || exit

  rm -rf ./run_mass_$DEVICE_ID
  mkdir ./run_mass_$DEVICE_ID

  cp train.py ./run_mass_$DEVICE_ID
  cp eval.py ./run_mass_$DEVICE_ID
  cp $configurations ./run_mass_$DEVICE_ID

  if [ $vocab ]
  then
    cp $vocab ./run_mass_$DEVICE_ID
  fi

  cd ./run_mass_$DEVICE_ID || exit
  env > log.log
  echo $task
  if [ "$task" == "train" ]
  then
    python train.py --config ${configurations##*/} >>log.log 2>&1 &
  elif [ "$task" == "infer" ]
  then
    python eval.py --config ${configurations##*/} --output ${output} --vocab ${vocab##*/} >>log_infer.log 2>&1 &
  fi
  cd ../
done
