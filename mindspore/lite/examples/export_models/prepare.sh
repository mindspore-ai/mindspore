#!/bin/bash

display_usage()
{
  echo "Usage: prepare.sh [-d mindspore_docker] [-c model_config_file] [-i]"
  echo "Options:"
  echo "    -d docker where mindspore is installed. If no docker is provided script will use local python"
  echo "    -c network configuration file. default is models_train.cfg"
  echo "    -i create input and output files"

}

checkopts()
{
  DOCKER=""
  TRAIN_IO=""
  CONFIG_FILE="models_train.cfg"
  while getopts 'c:d:i' opt
  do
    case "${opt}" in
      c)
        CONFIG_FILE=$OPTARG
        ;;
      d)
        DOCKER=$OPTARG
        ;;
      i)
        TRAIN_IO="train_io/"
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}

function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        if [ ! -z "${arr[0]}" ]; then
          printf "%-8s %-10s %-40s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
        fi
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}
export_result_file=export_result.txt
echo ' ' > ${export_result_file}


CLOUD_MODEL_ZOO=../../../../model_zoo/

checkopts "$@"

if [ -z "${DOCKER}" ]; then
    echo "MindSpore docker was not provided, attempting to run locally"
fi

mkdir -p mindir
if [ ! -z "${TRAIN_IO}" ]; then
  mkdir -p ${TRAIN_IO}
fi

while read line; do
    LFS=" " read -r -a line_array <<< ${line}
    model_name=${line_array[0]}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo 'exporting' ${model_name}
    if [ ! -z "${DOCKER}" ]; then
        docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER} /bin/bash -c "CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} PYTHONPATH=${CLOUD_MODEL_ZOO} python models/${model_name}_train_export.py ${TRAIN_IO} && chmod 444 mindir/${model_name}_train.mindir"
    else
        PYTHONPATH=${CLOUD_MODEL_ZOO} python models/${model_name}_train_export.py ${TRAIN_IO}
    fi
    if [ $? = 0 ]; then
      export_result='export mindspore '${model_name}'_train_export pass';echo ${export_result} >> ${export_result_file}
    else
      export_result='export mindspore '${model_name}'_train_export failed';echo ${export_result} >> ${export_result_file}
    fi
done < ${CONFIG_FILE}

Print_Result ${export_result_file}
rm ${export_result_file}
