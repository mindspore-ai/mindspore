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
    local cfg_file_list=("$models_gpu_fp32_config" "$models_gpu_weightquant_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $gpu_fail_not_return
}

# Run on gpu platform:
function Run_gpu() {
    # Prepare the config file list
    local gpu_cfg_file_list=("$models_gpu_fp32_config" "$models_gpu_fp16_config" "$models_gpu_weightquant_config")
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${gpu_cfg_file_list[*]}" . '/data/local/tmp' $run_gpu_log_file $run_benchmark_result_file 'arm64' 'GPU' $device_id $gpu_fail_not_return
}


function Run_mindrt_parallel() {
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi

        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        limit=`echo ${line} | awk -F ';' '{print $2}'`
        fp16=`echo ${line} | awk -F ';' '{print $3}'`

        data_path="/data/local/tmp/input_output/"
        output=${data_path}'output/'${model_name}'.ms.out'
        input=${data_path}'input/'${model_name}'.ms.bin'
        model=${model_name}'.ms'
        echo ${model_name} >> "${run_parallel_log_file}"
        echo "run mindrt parallel test : ${model_name}"

        ########## RUN CPU-CPU parallel
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt

        echo './benchmark --enableParallel=true --enableFp16='${fp16}' --accuracyThreshold='${limit}' --modelFile='${model}' --inDataFile='${input}' --benchmarkDataFile='${output} >> adb_run_cmd.txt
        echo './benchmark --enableParallel=true --enableFp16='${fp16}' --accuracyThreshold='${limit}' --modelFile='${model}' --inDataFile='${input}' --benchmarkDataFile='${output} >> "${run_parallel_log_file}"

        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_parallel_log_file}"
        if [ $? = 0 ]; then
            run_result='mindrt_parallel_CPU_CPU: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='mindrt_parallel_CPU_CPU: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            if [[ $gpu_fail_not_return != "ON" ]]; then
                return 1
            fi
        fi

        ########## RUN CPU-GPU parallel
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt

        echo './benchmark --enableParallel=true --device=GPU --enableFp16='${fp16}' --accuracyThreshold='${limit}' --modelFile='${model}' --inDataFile='${input}' --benchmarkDataFile='${output} >> adb_run_cmd.txt
        echo './benchmark --enableParallel=true --device=GPU --enableFp16='${fp16}' --accuracyThreshold='${limit}' --modelFile='${model}' --inDataFile='${input}' --benchmarkDataFile='${output} >> "${run_parallel_log_file}"

        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_parallel_log_file}"
        if [ $? = 0 ]; then
            run_result='mindrt_parallel_CPU_GPU: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='mindrt_parallel_CPU_GPU: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            if [[ $gpu_fail_not_return != "ON" ]]; then
                return 1
            fi
        fi
    done < ${models_mindrt_parallel_config}
}

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_benchmark_gpu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
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
            gpu_fail_not_return=${OPTARG}
            echo "gpu_fail_not_return is ${OPTARG}"
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
models_gpu_fp32_config=${basepath}/../config/models_gpu_fp32.cfg
models_gpu_fp16_config=${basepath}/../config/models_gpu_fp16.cfg
models_gpu_weightquant_config=${basepath}/../config/models_gpu_weightquant_8bit.cfg
models_mindrt_parallel_config=${basepath}/../config/models_mindrt_parallel.cfg

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

run_gpu_log_file=${basepath}/run_gpu_log.txt
echo 'run gpu logs: ' > ${run_gpu_log_file}
run_parallel_log_file=${basepath}/run_parallel_log.txt
echo 'run parallel logs: ' > ${run_parallel_log_file}

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
if [[ $backend == "all" || $backend == "gpu" ]]; then
    # Run on gpu
    echo "start Run gpu ..."
    Run_gpu
    Run_gpu_status=$?
    # Run_gpu_PID=$!
    # sleep 1
fi
# guard mindrt parallel
if [[ $backend == "all" || $backend == "gpu" || $backend == "mindrt_parallel" ]]; then
    echo "start Run Mindrt Parallel ... "
    Run_mindrt_parallel
    Run_mindrt_parallel_status=$?
    # Run_mindrt_parallel_PID=$!
    # sleep 1
fi
# guard cropper
if [[ $backend == "all" || $backend == "gpu" || $backend == "cropper" ]]; then
    echo "start Run Cropper ... "
    cd ${basepath} || exit 1
    bash ${basepath}/scripts/run_cropper.sh -r ${release_path} -d ${device_id}
    Run_cropper_status=$?
    # Run_cropper_PID=$!
    # sleep 1
fi

if [[ $backend == "all" || $backend == "gpu" ]]; then
    # wait ${Run_gpu_PID}
    # Run_gpu_status=$?
    if [[ ${Run_gpu_status} != 0 ]];then
        echo "Run_gpu failed"
        cat ${run_gpu_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "gpu" || $backend == "mindrt_parallel" ]]; then
    # wait ${Run_mindrt_parallel_PID}
    # Run_mindrt_parallel_status=$?
    if [[ ${Run_mindrt_parallel_status} != 0 ]];then
        echo "Run_mindrt_parallel failed"
        cat ${run_parallel_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "gpu" || $backend == "cropper" ]]; then
    # wait ${Run_cropper_PID}
    # Run_cropper_status=$?
    if [[ ${Run_cropper_status} != 0 ]];then
        echo "Run cropper failed"
        cat ${run_gpu_log_file}
        isFailed=1
    fi
fi

echo "Run_gpu and Run_cropper and mindrt_parallel is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
