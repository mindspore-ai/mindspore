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
    local npu_cfg_file_list=("$models_npu_config" "$models_npu_weightquant_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${npu_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $npu_fail_not_return
}

# Run on npu platform:
function Run_npu() {
    # Prepare the config file list
    local npu_cfg_file_list=("$models_npu_config" "$models_npu_fp16_config" "$models_npu_weightquant_config")
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${npu_cfg_file_list[*]}" . '/data/local/tmp' $run_npu_log_file $run_benchmark_result_file 'arm64' 'NPU' $device_id $npu_fail_not_return
}

basepath=$(pwd)
echo ${basepath}

# Example:sh run_benchmark_npu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:p:" opt; do
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
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        p)
            npu_fail_not_return=${OPTARG}
            echo "npu_fail_not_return is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# mkdir train
x86_path=${release_path}/ubuntu_x86
arm64_path=${release_path}/android_aarch64/npu
file_name=$(ls ${x86_path}/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_npu_config=${basepath}/../config/models_npu.cfg
models_npu_fp16_config=${basepath}/../config/models_npu_fp16.cfg
models_npu_weightquant_config=${basepath}/../config/models_npu_weightquant_8bit.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
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

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_npu_log_file=${basepath}/run_npu_log.txt
echo 'run npu logs: ' > ${run_npu_log_file}

# Copy the MindSpore models:
echo "Push files to the arm and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

# Push files to the phone
Push_Files $arm64_path "aarch64" $version $benchmark_test_path "adb_push_log.txt" $device_id

backend=${backend:-"all"}
isFailed=0
if [[ $backend == "all" || $backend == "npu" || $backend == "npu_fp32" ]]; then
    # Run on npu
    echo "start Run npu ..."
    Run_npu
    Run_npu_status=$?
    # Run_npu_PID=$!
    # sleep 1
fi

if [[ $backend == "all" || $backend == "npu" || $backend == "npu_fp32" ]]; then
    # wait ${Run_npu_PID}
    # Run_npu_status=$?
    if [[ ${Run_npu_status} != 0 ]];then
        echo "Run_npu failed"
        cat ${run_npu_log_file}
        isFailed=1
    fi
fi

echo "Run_npu ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
