#!/bin/bash

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

    # Convert npu models:
    while read line; do
        if [[ $line == \#* ]]; then
          continue
        fi
        model_info=`echo ${line}|awk -F ' ' '{print $1}'`
        model_name=${model_info%%;*}
        model_type=${model_name##*.}
        case $model_type in
          pb)
            model_fmk="TF"
            ;;
          tflite)
            model_fmk="TFLITE"
            ;;
          onnx)
            model_fmk="ONNX"
            ;;
          mindir)
            model_fmk="MINDIR"
            ;;
          *)
            model_type="caffe"
            model_fmk="CAFFE"
            ;;
        esac
        if [[ $model_fmk == "CAFFE" ]]; then
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='$models_path/${model_name}'.prototxt --weightFile='$models_path'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name} >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}
        else
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name} >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}
        fi
        if [ $? = 0 ]; then
            converter_result='converter npu '${model_type}' '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter npu '${model_type}' '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_npu_config}
}

# Run on npu platform:
function Run_npu() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run npu converted models:
    while read line; do
        model_line_info=${line}
        if [[ $model_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${line}|awk -F ' ' '{print $2}'`
        input_num=`echo ${line}|awk -F ' ' '{print $3}'`
        data_path="/data/local/tmp/input_output/"
        input_files=''
        if [[ -z "$input_num" || $input_num == 1 ]]; then
          input_files=${data_path}'input/'$model_name'.ms.bin'
        elif [[ ! -z "$input_num" && $input_num -gt 1 ]]; then
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
          done
        fi
        echo "mindspore run npu: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_npu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=NPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_npu_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=NPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_npu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_npu: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_npu: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_npu_config}
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
#set -e

# Example:sh run_benchmark_npu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
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
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
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

# mkdir train

x86_path=${release_path}/ubuntu_x86
file_name=$(ls ${x86_path}/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_npu_config=${basepath}/../config/models_npu.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?

# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Converter_Result
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result
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

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "npu" ]]; then
    # Run on npu
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run npu ..."
    Run_npu
    Run_npu_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "npu" ]]; then
    if [[ ${Run_npu_status} != 0 ]];then
        echo "Run_npu failed"
        cat ${run_npu_log_file}
        isFailed=1
    fi
fi

# guard cropper
if [[ $backend == "all" || $backend == "npu" ]]; then
    cd ${basepath} || exit 1
    bash ${basepath}/scripts/run_cropper.sh -r ${release_path} -d ${device_id}
    Run_cropper_status=$?
    if [[ ${Run_cropper_status} != 0 ]];then
        echo "Run cropper failed"
        cat ${run_npu_log_file}
        isFailed=1
        exit 1
    fi
fi

echo "Run_npu and Run_cropper is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
