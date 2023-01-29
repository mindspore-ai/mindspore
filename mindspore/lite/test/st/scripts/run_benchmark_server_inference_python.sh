#!/bin/bash

function Run_server_python_ST() {
  # $1:basePath; $2:whlPath; $3:modelPath; $4:cfgFileList; $5:backend;
  base_path=$1
  whl_path=$2
  model_path=$3
  in_data_path=$4
  cfg_file_list=$5
  backend=$6
  mindspore_lite_whl=`ls ${whl_path}/*.whl`
  if [[ -f "${mindspore_lite_whl}" ]]; then
    pip install ${mindspore_lite_whl} --force-reinstall || exit 1
    echo "install python whl success."
  else
    echo "not find python whl.."
    exit 1
  fi
  echo "Run_python st..."
  echo "-----------------------------------------------------------------------------------------"
  cd ${base_path}/python/ || exit 1
  for cfg_file in ${cfg_file_list[*]}; do
    while read line; do
      line_info=${line}
      if [[ $line_info == \#* || $line_info == "" ]]; then
        continue
      fi
      model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
      input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
      input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
      input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
      input_files=""
      data_path=${in_data_path}"/input_output/"
      if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
        input_files=${data_path}'input/'${model_name}'.ms.bin'
      else
        for i in $(seq 1 $input_num)
        do
          input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
        done
      fi
      model_file=${model_path}'/'${model_name}'.ms'
      python test_server_inference.py ${model_file} ${input_files} ${input_shapes} ${backend}
      Run_python_st_status=$?
      if [[ ${Run_python_st_status} != 0 ]];then
        echo "run python model name:     ${model_name}     failed.";
        echo "Run_python_st_status failed"
        exit 1
      fi
      echo "run python model name:     ${model_name}     pass.";
    done < ${cfg_file}
  done
  echo "-----------------------------------------------------------------------------------------"
}
