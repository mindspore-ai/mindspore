#!/bin/bash
source ./scripts/base_functions.sh

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Prepare the config file list
    local x86_asan_cfg_file_list=("$models_asan_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${x86_asan_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file
}

function Run_x86_asan() {
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-x64' >> "${run_x86_asan_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1
    # Prepare the config file list
    local asan_cfg_file_list=("$models_asan_config")
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:benchmark_mode
    Run_Benchmark "${asan_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_asan_log_file $run_benchmark_result_file 'x86' 'CPU' ''
}

basepath=$(pwd)
echo ${basepath}

# Example:sh run_benchmark_x86.sh -r /home/temp_test -m /home/temp_test/models -e arm_cpu
while getopts "r:m:e:l:" opt; do
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
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

x86_path=${release_path}/centos_x86/asan
cd ${x86_path}
file_name=$(ls *-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_asan_config=${basepath}/../${config_folder}/models_asan.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter &
Run_converter_PID=$!
sleep 2

wait ${Run_converter_PID}
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
# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_x86_asan_log_file=${basepath}/run_x86_asan_log.txt
echo 'run x86 asan logs: ' > ${run_x86_asan_log_file}

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "x86_asan" ]]; then
    # Run on x86
    echo "start Run x86 ASAN..."
    Run_x86_asan &
    Run_x86_asan_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86_asan" ]]; then
    wait ${Run_x86_asan_PID}
    Run_x86_asan_status=$?

    # Check benchmark result and return value
    if [[ ${Run_x86_asan_status} != 0 ]];then
        echo "Run_x86 ASAN failed"
        cat ${run_x86_asan_log_file}
        isFailed=1
    fi
fi

echo "Run_x86_asan is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
