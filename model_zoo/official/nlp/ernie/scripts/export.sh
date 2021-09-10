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
if [ $# -ne 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh export.sh CKPT_FILE EXPORT_PATH TASK_TYPE"
    echo "for example: sh sh export.sh /path/ckpt.ckpt /path/ msra_ner"
    echo "TASK_TYPE including msra_ner, chnsenticorp"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
exit 1
fi
ulimit -u unlimited
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
CKPT_FILE=$(get_real_path $1)
echo $CKPT_FILE
if [ ! -f $CKPT_FILE ]
then
    echo "error: CKPT_FILE=$CKPT_FILE is not valid"
exit 1
fi
EXPORT_PATH=$(get_real_path $2)
echo $EXPORT_PATH
if [ ! -d $EXPORT_PATH ]
then
    echo "error: EXPORT_PATH=$EXPORT_PATH is not valid"
exit 1
fi

TASK_TYPE=$3
NUMBER_LABELS=3
EVAL_BATCH_SIZE=1
case $TASK_TYPE in
  "msra_ner")
    NUMBER_LABELS=7
    ;;
  "dbqa")
    NUMBER_LABELS=2
    ;;
  esac

CUR_DIR=`pwd`
python ${CUR_DIR}/export.py \
        --task_type=${TASK_TYPE} \
        --device_id=0 \
        --batch_size=${EVAL_BATCH_SIZE} \
        --number_labels=${NUMBER_LABELS} \
        --ckpt_file="${CKPT_FILE}" \
        --file_name="${EXPORT_PATH}/${TASK_TYPE}.mindir" \
        --file_format="MINDIR" \
        --device_target="Ascend"
