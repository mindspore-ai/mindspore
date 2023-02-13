#!/bin/bash
source ./scripts/base_functions.sh
source ./scripts/run_benchmark_python.sh

# Run converter on x86 platform:
function Run_Converter() {
    cd ${path}/server || exit 1
    if [[ $backend == "all" || $backend == "server_inference_x86" ]]; then
        tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
        cd ${path}/server/mindspore-lite-${version}-linux-x64/ || exit 1
    fi
    if [[ $backend == "all" || $backend == "server_inference_arm" ]]; then
        tar -zxf mindspore-lite-${version}-linux-aarch64.tar.gz || exit 1
        cd ${path}/server/mindspore-lite-${version}-linux-aarch64/ || exit 1
    fi
    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}
    echo ${models_server_inference_cfg_file_list[*]}

    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${models_server_inference_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $run_fail_not_return
    convert_status=$?
    if [[ convert_status -ne 0 ]]; then
      echo "run server inference convert failed."
      return 1
    fi
}

function Run_server_inference_avx512() {
    cd ${path}/server || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${path}/server/mindspore-lite-${version}-linux-x64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:benchmark_mode
    Run_Benchmark "${models_server_inference_cfg_file_list[*]}" $ms_models_path $models_path $run_server_inference_x86_log_file $run_benchmark_result_file 'x86_avx512' 'CPU' '' $run_fail_not_return
    run_benchmark_status=$?
    if [[ run_benchmark_status -ne 0 ]]; then
      echo "run server inference benchmark failed."
      return 1
    fi
}

# Run on arm64 platform:
function Run_server_inference_arm64() {
    cd ${path}/server || exit 1
    tar -zxf mindspore-lite-${version}-linux-aarch64.tar.gz || exit 1
    cd ${path}/server/mindspore-lite-${version}-linux-aarch64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:benchmark_mode
    Run_Benchmark "${models_server_inference_cfg_file_list[*]}" $ms_models_path $models_path $run_server_inference_arm64_log_file $run_benchmark_result_file 'x86_avx512' 'CPU' '' $run_fail_not_return
    run_benchmark_status=$?
    if [[ run_benchmark_status -ne 0 ]]; then
      echo "run server inference benchmark failed."
      return 1
    fi
}

# Example:sh run_benchmark_gpu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:p:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        p)
            run_fail_not_return=${OPTARG}
            echo "run_fail_not_return is ${OPTARG}"
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

basepath=$(pwd)
path=${release_path}
file_name=""
if [[ $backend == "all" || $backend == "server_inference_x86" ]]; then
    path=${path}/centos_x86
    cd ${path}/server || exit 1
    file_name=$(ls *linux-x64.tar.gz)
fi
if [[ $backend == "all" || $backend == "server_inference_arm" ]]; then
    path=${path}/linux_aarch64
    cd ${path}/server || exit 1
    file_name=$(ls *linux-aarch64.tar.gz)
fi
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd ${basepath}

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_server_inference_config=${basepath}/../${config_folder}/models_server_inference.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

models_server_inference_cfg_file_list=()
models_server_inference_cfg_file_list=("$models_server_inference_config")

# Run converter
echo "start Run converter ..."
Run_Converter
Run_converter_status=$?
# Check converter result and return value
Print_Converter_Result $run_converter_result_file

if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    exit 1
fi
# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_server_inference_x86_log_file=${basepath}/run_server_inference_x86_log.txt
echo 'run server inference x86 logs: ' > ${run_server_inference_x86_log_file}
run_server_inference_arm64_log_file=${basepath}/run_server_inference_arm_log.txt
echo 'run server inference arm64 logs: ' > ${run_server_inference_arm64_log_file}

# Copy the MindSpore models:
echo "Push files and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

backend=${backend:-"all"}
isFailed=0
if [[ $backend == "all" || $backend == "server_inference_x86" ]]; then
    echo "start Run ..."
    Run_server_inference_avx512
    Run_x86_status=$?
fi

if [[ $backend == "all" || $backend == "server_inference_arm" ]]; then
    echo "start Run ... "
    Run_server_inference_arm64
    Run_arm64_status=$?
fi


if [[ $backend == "all" || $backend == "server_inference_x86" ]]; then
    if [[ ${Run_x86_status} != 0 ]];then
        echo "run x86 server inference failed"
        cat ${run_server_inference_x86_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "server_inference_arm" ]]; then
    if [[ ${Run_arm64_status} != 0 ]];then
        echo "run arm64 server inference failed"
        cat ${run_server_inference_arm64_log_file}
        isFailed=1
    fi
fi

# run python ST
if [[ $backend == "all" || $backend == "server_inference_x86" || $backend == "server_inference_arm" ]]; then
  models_python_config=${basepath}/../config_level0/models_server_inference_python.cfg
  models_python_cfg_file_list=("$models_python_config")
  Run_python_ST ${basepath} ${path}/server ${ms_models_path} ${models_path} "${models_python_cfg_file_list[*]}" "CPU_PARALLEL" ".ms"
  Run_python_status=$?
  if [[ ${Run_python_status} != 0 ]];then
      echo "Run_python_status failed"
      isFailed=1
  fi
fi

echo "run x86_server_inference and arm_server_inference is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
