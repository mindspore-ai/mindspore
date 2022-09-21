#!/bin/bash
source ./scripts/base_functions.sh

# Run converter for NNIE models on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../tools/converter/lib/:../tools/converter/third_party/glog/lib:../tools/converter/providers/Hi3516D/third_party/opencv-4.2.0:../tools/converter/providers/Hi3516D/third_party/protobuf-3.9.0
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../runtime/lib/

    echo ' ' > ${run_converter_log_file}
    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}
    cur_timestamp=$((`date '+%s'`*1000+10#`date '+%N'`/1000000))
    fail_status_file=${cur_timestamp}.log
    echo '0' > ${fail_status_file}

    # parallel processing
    fifo_file="fifo_file.txt"
    mkfifo ${fifo_file}
    exec 6<>${fifo_file}
    rm -f ${fifo_file}
    max_parallel_jobs=4
    if [[ ${level} == "level1" ]]; then
      max_parallel_jobs=1
    fi
    for ((i = 0; i < ${max_parallel_jobs}; i++)); do echo; done >&6

    # Convert nnie models:
    while read line; do
        nnie_line_info=${line}
        if [[ $nnie_line_info == \#* || $nnie_line_info == "" ]]; then
          continue
        fi
        read -u6
        {
          model_location=`echo ${nnie_line_info}|awk -F ' ' '{print $1}'`
          model_info=`echo ${nnie_line_info}|awk -F ' ' '{print $2}'`
          model_name=${model_info%%;*}

          # generate converter config file
          mkdir -p ./${model_name} || exit 1
          cd ./${model_name} || exit 1
          cp ../tools/converter/converter/converter_lite ./ || exit 1
          cp ${models_path}/${model_location}/${model_name}.cfg ./ || exit 1

          ms_config_file=./converter.cfg
          echo "[registry]" > ${ms_config_file}
          echo 'plugin_path='${x86_path}'/mindspore-lite-'${version}'-linux-x64/tools/converter/providers/Hi3516D/libmslite_nnie_converter.so' >> ${ms_config_file}
          echo '[nnie]' >> ${ms_config_file}
          echo 'nnie_mapper_path=../tools/converter/providers/Hi3516D/libnnie_mapper.so' >> ${ms_config_file}
          echo 'nnie_data_process_path=../tools/converter/providers/Hi3516D/libmslite_nnie_data_process.so' >> ${ms_config_file}
          echo 'benchmark_path='${x86_path}'/mindspore-lite-'${version}'-linux-x64/tools/benchmark/benchmark' >> ${ms_config_file}
          echo 'nnie_config_path='./${model_name}.cfg >> ${ms_config_file}
          echo -e 'nnie_disable_inplace_fusion=off\n' >> ${ms_config_file}
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_location}'/model/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_location}'/model/'${model_name}'.caffemodel --configFile='${ms_config_file}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_location}/model/${model_name}.prototxt --weightFile=${models_path}/${model_location}/model/${model_name}.caffemodel --configFile=${ms_config_file} --outputFile=${ms_models_path}/${model_name}
          if [ $? = 0 ]; then
              rm -rf ${x86_path}/mindspore-lite-${version}-linux-x64/${model_name}
              converter_result='converter CAFFE '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
          else
              rm -rf ${x86_path}/mindspore-lite-${version}-linux-x64/${model_name}
              converter_result='converter CAFFE '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};
              echo '1' > ${fail_status_file}
          fi
          echo >&6
        } &
    done < ${models_nnie_config}
    wait
    exec 6>&-
    read fail_status < ${fail_status_file}
    return ${fail_status}
}

# Run benchmark on hi3516:
function Run_Hi3516() {
  cd ${arm32_path} || exit 1
  tar -zxf mindspore-lite-${version}-linux-aarch32.tar.gz || exit 1
  cd ${arm32_path}/mindspore-lite-${version}-linux-aarch32 || return 1

  chmod +x ./tools/benchmark/benchmark
  # copy related files to benchmark_test
  cp -a ./providers/Hi3516D/libmslite_nnie.so ${benchmark_test_path}/libmslite_nnie.so || exit 1
  cp -a ./providers/Hi3516D/libmslite_proposal.so ${benchmark_test_path}/libmslite_proposal.so || exit 1
  cp -a ./tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1
  cp -a ./runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1

  # cp files to nfs shared folder
  echo "start push files to hi3516"
  echo ${device_ip}
  scp ${benchmark_test_path}/* root@${device_ip}:/user/nnie/benchmark_test/ || exit 1
  ssh root@${device_ip} "cd /user/nnie/benchmark_test; sh run_benchmark_nnie.sh"
  if [ $? = 0 ]; then
    run_result='hi3516: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file};
  else
    run_result='hi3516: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; exit 1
  fi
}

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_nnie_nets.sh r /home/temp_test -m /home/temp_test/models -e arm32_3516D -d 192.168.1.1
while getopts "r:m:d:e:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_ip=${OPTARG}
            echo "device_ip is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

x86_path=${release_path}/centos_x86
arm32_path=${release_path}/linux_aarch32

file_name=$(ls ${x86_path}/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_nnie_config=${basepath}/../${config_folder}/models_nnie.cfg
run_hi3516_script=${basepath}/scripts/nnie/run_benchmark_nnie.sh

# Set ms models output path
ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter for NNIE models..."
Run_Converter &
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter for NNIE models success"
    Print_Converter_Result ${run_converter_result_file}
else
    echo "Run converter for NNIE models failed"
    cat ${run_converter_log_file}
    Print_Converter_Result ${run_converter_result_file}
    exit 1
fi

if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

# Copy the MindSpore models:
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
cp -a ${models_nnie_config} ${benchmark_test_path} || exit 1
cp -a ${run_hi3516_script} ${benchmark_test_path} || exit 1

if [[ $backend == "all" || $backend == "arm32_3516D" ]]; then
    # Run on hi3516
    file_name=$(ls ${arm32_path}/*linux-aarch32.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    Run_Hi3516 &
    Run_hi3516_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm32_3516D" ]]; then
    wait ${Run_hi3516_PID}
    Run_hi3516_status=$?
    # Check benchmark result and return value
    if [[ ${Run_hi3516_status} != 0 ]];then
        echo "Run_hi3516 failed"
        isFailed=1
    else
        echo "Run_hi3516 success"
        isFailed=0
    fi
fi

if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
