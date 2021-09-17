#!/bin/bash

# Run on NVIDIA TensorRT platform:
function Run_TensorRT() {
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
    source /etc/profile

    while read line; do
        model_name=${line%;*}
        # length=${#model_name}
        # input_shapes=${line:length+1}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_tensorrt_log_file}"
        # inputshape needed later
        echo 'CUDA_VISILE_DEVICE='${cuda_device_id}' ./benchmark --modelFile='${basepath}'/'${model_name}'.ms --inDataFile='${basepath}'/../../input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${basepath}'/../../input_output/output/'${model_name}'.ms.out --device=GPU' >> "${run_tensorrt_log_file}"
        CUDA_VISILE_DEVICE=${cuda_device_id} ./benchmark --modelFile=$basepath/${model_name}.ms --inDataFile=${basepath}/../../input_output/input/${model_name}.ms.bin --benchmarkDataFile=${basepath}/../../input_output/output/${model_name}.ms.out --device=GPU >> ${run_tensorrt_log_file}
        if [ $? = 0 ]; then
            run_result='TensorRT: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='TensorRT: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tensorrt_config}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                    Testcase                                                                                           Result   "
    echo -e "---                    --------                                                                                           ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo "on tensorrt device, bashpath is ${basepath}"

# Example:sh run_benchmark_tensorrt.sh -d 0
while getopts "d:" opt; do
    case ${opt} in
        d)
            cuda_device_id=${OPTARG}
            echo "cuda_device_id is ${cuda_device_id}."
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# Set models config filepath
models_tensorrt_config=${basepath}/models_tensorrt.cfg
echo ${models_tensorrt_config}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_tensorrt_log_file=${basepath}/run_tensorrt_log.txt
echo 'run tensorrt logs: ' > ${run_tensorrt_log_file}

echo "Running in tensorrt on device ${cuda_device_id} ..."
Run_TensorRT &
Run_TensorRT_PID=$!
sleep 1

wait ${Run_TensorRT_PID}
Run_benchmark_status=$?

# Check converter result and return value
echo "Run x86 TensorRT GPU ended on device ${cuda_device_id}"
Print_Benchmark_Result $run_benchmark_result_file
exit ${Run_benchmark_status}