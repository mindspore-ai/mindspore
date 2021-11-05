#!/bin/bash
source ./scripts/base_functions.sh

# Run converter for tensorrt x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf ${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Prepare the config file list
    local tensorrt_cfg_file_list=("$models_tensorrt_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${tensorrt_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $x86_fail_not_return
}

function Run_Tensorrt() {
  # Unzip x86 runtime and converter
  cd ${x86_path}/tensorrt || exit 1
  tar -zxf ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
  tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
  cd ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64/ || exit 1

  rm -rf ${benchmark_test_path}
  mkdir -p ${benchmark_test_path}

  cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
  cp -a ${models_tensorrt_config} ${benchmark_test_path} || exit 1
  cp -a ${run_tensorrt_benchmark_script} ${benchmark_test_path} || exit 1

  chmod +x ./tools/benchmark/benchmark
  # copy related files to benchmark_test
  cp -a ./tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1
  cp -a ./runtime/lib/lib*.so* ${benchmark_test_path}/ || exit 1

  echo "start push files to nvidia device ${device_ip} : ${cuda_device_id}"
  ssh tensorrt@${device_ip} "cd ${device_benchmark_test_path}; rm -rf ./*"
  scp ${benchmark_test_path}/* tensorrt@${device_ip}:${device_benchmark_test_path} || exit 1
  ssh tensorrt@${device_ip} "cd ${device_benchmark_test_path}; sh run_benchmark_tensorrt.sh -d ${cuda_device_id}"
  if [ $? = 0 ]; then
    run_result='run tensorrt on device: '${cuda_device_id}' pass'; echo ${run_result} >> ${run_benchmark_result_file};
  else
    run_result='run tensorrt on device: '${cuda_device_id}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; exit 1
  fi
}

# Example:sh run_benchmark_tensorrt.sh -r /home/temp_test -m /home/temp_test/models -e x86_gpu -d 192.168.1.1:0
while getopts "r:m:d:e:" opt; do
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
            device_ip=`echo ${OPTARG} | cut -d \: -f 1`
            cuda_device_id=`echo ${OPTARG} | cut -d \: -f 2`
            echo "device_ip is ${device_ip}, cuda_device_id is ${cuda_device_id}."
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done
x86_fail_not_return="OFF" # This value could not be set to ON.
basepath=$(pwd)/${cuda_device_id}
rm -rf ${basepath}
mkdir -p ${basepath}
echo "NVIDIA TensorRT, bashpath is ${basepath}"

x86_path=${release_path}/ubuntu_x86
file_name=$(ls ${x86_path}/*-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
ms_models_path=${basepath}/ms_models

# Set models config filepath
models_tensorrt_config=${basepath}/../../config/models_tensorrt.cfg
run_tensorrt_benchmark_script=${basepath}/../scripts/tensorrt/run_benchmark_tensorrt.sh
echo ${models_tensorrt_config}

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "Start run converter in tensorrt ..."
Run_Converter
Run_converter_status=$?

# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit 1
fi

# push benchmark, so, and ms models to nvidia device
benchmark_test_path=${basepath}/benchmark_test
device_benchmark_test_path=/home/tensorrt/benchmark_test/${cuda_device_id}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

echo "Start run benchmark in tensorrt ..."
Run_Tensorrt &
Run_Tensorrt_PID=$!
sleep 1

wait ${Run_Tensorrt_PID}
Run_benchmark_status=$?

run_tensorrt_log_file=${basepath}/run_tensorrt_log.txt
scp tensorrt@${device_ip}:${device_benchmark_test_path}/run_tensorrt_log.txt ${run_tensorrt_log_file} || exit 1

echo "Run x86 TensorRT GPU ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${Run_benchmark_status}
