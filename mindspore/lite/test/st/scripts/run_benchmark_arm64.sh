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
    local fp32_cfg_file_list=("$models_tf_config" "$models_tflite_config" "$models_caffe_config" "$models_onnx_config" "$models_mindspore_config" \
                              "$models_mindspore_train_config" "$models_posttraining_config" "$models_process_only_fp16_config" \
                              "$models_tflite_awaretraining_config" "$models_weightquant_config" "$models_weightquant_7bit_config" \
                              "$models_weightquant_9bit_config" "$models_process_only_config")

    local fp16_cfg_file_list=("$models_onnx_fp16_config" "$models_caffe_fp16_config" "$models_tflite_fp16_config" "$models_tf_fp16_config")
    # Convert models:
    if [[ $1 == "all" || $1 == "arm64_cpu" || $1 == "arm64_fp32" ]]; then
        # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
        Convert "${fp32_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $arm64_fail_not_return
    fi
    if [[ $1 == "arm64_fp16" ]]; then
        Convert "${fp16_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $arm64_fail_not_return
    fi
}

# Run on arm64 platform:
function Run_arm64() {
    # Prepare the config file list
    local arm64_cfg_file_list=("$models_tf_config" "$models_tflite_config" "$models_caffe_config" "$models_onnx_config" "$models_mindspore_config" \
                              "$models_mindspore_train_config" "$models_posttraining_config" "$models_compatibility_config" \
                              "$models_tflite_awaretraining_config" "$models_weightquant_config" "$models_weightquant_7bit_config" \
                              "$models_weightquant_9bit_config" "$models_process_only_config" "$models_process_only_fp16_config")
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${arm64_cfg_file_list[*]}" . '/data/local/tmp' $run_arm64_fp32_log_file $run_benchmark_result_file 'arm64' 'CPU' $device_id $arm64_fail_not_return
}

# Run on arm64-fp16 platform:
function Run_arm64_fp16() {
    local arm64_cfg_file_list=("$models_onnx_fp16_config" "$models_caffe_fp16_config" "$models_tflite_fp16_config" "$models_tf_fp16_config" \
                               "$models_process_only_fp16_config")
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${arm64_cfg_file_list[*]}" . '/data/local/tmp' $run_arm64_fp16_log_file $run_benchmark_result_file 'arm64' 'CPU' $device_id $arm64_fail_not_return
}

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_benchmark_arm.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
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
            arm64_fail_not_return=${OPTARG}
            echo "arm64_fail_not_return is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# package info
x86_path=${release_path}/ubuntu_x86
arm64_path=${release_path}/android_aarch64/npu
file_name=$(ls ${x86_path}/*-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_tflite_config=${basepath}/../config/models_tflite.cfg
models_tf_config=${basepath}/../config/models_tf.cfg
models_caffe_config=${basepath}/../config/models_caffe.cfg
models_tflite_awaretraining_config=${basepath}/../config/models_tflite_awaretraining.cfg
models_posttraining_config=${basepath}/../config/models_posttraining.cfg
models_onnx_config=${basepath}/../config/models_onnx.cfg
models_onnx_fp16_config=${basepath}/../config/models_onnx_fp16.cfg
models_caffe_fp16_config=${basepath}/../config/models_caffe_fp16.cfg
models_tflite_fp16_config=${basepath}/../config/models_tflite_fp16.cfg
models_tf_fp16_config=${basepath}/../config/models_tf_fp16.cfg
models_mindspore_config=${basepath}/../config/models_mindspore.cfg
models_mindspore_train_config=${basepath}/../config/models_mindspore_train.cfg
models_weightquant_7bit_config=${basepath}/../config/models_weightquant_7bit.cfg
models_weightquant_9bit_config=${basepath}/../config/models_weightquant_9bit.cfg
models_weightquant_config=${basepath}/../config/models_weightquant.cfg
models_compatibility_config=${basepath}/../config/models_compatibility.cfg
models_process_only_config=${basepath}/../config/models_process_only.cfg
models_process_only_fp16_config=${basepath}/../config/models_process_only_fp16.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

backend=${backend:-"all"}

# Run converter
echo "start Run converter ..."
Run_Converter $backend
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

run_arm64_fp32_log_file=${basepath}/run_arm64_fp32_log.txt
echo 'run arm64_fp32 logs: ' > ${run_arm64_fp32_log_file}
run_arm64_fp16_log_file=${basepath}/run_arm64_fp16_log.txt
echo 'run arm64_fp16 logs: ' > ${run_arm64_fp16_log_file}

# Copy the MindSpore models:
echo "Push files to the arm and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
# Copy models converted using old release of mslite converter for compatibility test
cp -a ${models_path}/compatibility_test/*.ms ${benchmark_test_path} || exit 1
# Push files to the phone
Push_Files $arm64_path "aarch64" $version $benchmark_test_path "adb_push_log.txt" $device_id

backend=${backend:-"all"}
isFailed=0
if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp32" ]]; then
    # Run on arm64
    echo "start Run arm64 ..."
    Run_arm64
    Run_arm64_fp32_status=$?
#    Run_arm64_fp32_PID=$!
#    sleep 1
fi
if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp16" ]]; then
    # Run on arm64-fp16
    echo "start Run arm64-fp16 ..."
    Run_arm64_fp16
    Run_arm64_fp16_status=$?
#    Run_arm64_fp16_PID=$!
#    sleep 1
fi

if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp32" ]]; then
#    wait ${Run_arm64_fp32_PID}
#    Run_arm64_fp32_status=$?
    if [[ ${Run_arm64_fp32_status} != 0 ]];then
        echo "Run_arm64_fp32 failed"
        cat ${run_arm64_fp32_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp16" ]]; then
#    wait ${Run_arm64_fp16_PID}
#    Run_arm64_fp16_status=$?
    if [[ ${Run_arm64_fp16_status} != 0 ]];then
        echo "Run_arm64_fp16 failed"
        cat ${run_arm64_fp16_log_file}
        isFailed=1
    fi
fi

echo "Run_arm64_fp32 and Run_arm64_fp16 is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
