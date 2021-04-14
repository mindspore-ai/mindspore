#!/bin/bash

function Run_x86() {
    local CODEGEN_PATH=${x86_path}/mindspore-lite-${version}-inference-linux-x64/tools/codegen

    rm -rf ${build_path}
    mkdir -p ${build_path}

    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        ${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms
        # 1. build benchmark
        mkdir -p ${build_path}/${model_name}/build && cd ${build_path}/${model_name}/build || exit 1
        cmake -DPKG_PATH=${x86_path}/mindspore-lite-${version}-inference-linux-x64 ${build_path}/${model_name}
        make
        # 2. run benchmark
        echo "net file: ${build_path}/${model_name}/src/net.bin"
        ./benchmark ${models_path}/input_output/input/${model_name}.ms.bin ${build_path}/${model_name}/src/net.bin 1 ${models_path}/input_output/output/${model_name}.ms.out
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}

    rm -rf ${build_path}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                  Testcase                                                                                             Result   "
    echo -e "---                  --------                                                                                             ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Converter_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        printf "%-15s %-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
    done < ${run_converter_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < ${run_benchmark_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo ${basepath}

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -s /home/temp_test/ms_models -d "8KE5T19620002408"
while getopts "r:m:e:s:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        s)
            ms_models_path=${OPTARG}
            echo "ms_models_path is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

x86_path=${release_path}/ubuntu_x86
file_name=$(ls ${x86_path}/*inference-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

build_path=${basepath}/build
models_tflite_config=${basepath}/models_tflite.cfg

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

run_x86_log_file=${basepath}/run_x86_log.txt
echo 'run x86 logs: ' > ${run_x86_log_file}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

backend=${backend:-"all"}
isFailed=0
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" ]]; then
    # Run on x86
    echo "start Run x86 ..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?

    # Check benchmark result and return value
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86 failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi

echo "Run_x86 is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
