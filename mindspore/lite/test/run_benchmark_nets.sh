#!/bin/bash

# Run on x86 platform:
function Run_x86() {
    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}
        echo 'cd  '${convertor_path}'/MSLite-*-linux_x86_64'
        cd ${convertor_path}/MSLite-*-linux_x86_64 || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' || return 1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1
        if [ $? = 0 ]; then
	    run_result='Run_x86: '${model_name}' pass'
	    echo ${run_result} >> ${run_benchmark_result_file}
	else
	    run_result='Run_x86: '${model_name}' fail <<===========================this is the failed case'
	    echo ${run_result} >> ${run_benchmark_result_file}
	    return 1
        fi
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}
        echo 'cd  '${convertor_path}'/MSLite-*-linux_x86_64'
        cd ${convertor_path}/MSLite-*-linux_x86_64 || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' || return 1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1
        if [ $? = 0 ]; then
            run_result='Run_x86: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='Run_x86: '${model_name}' fail <<===========================this is the failed case'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_caffe_config}
}

# Run on arm64 platform:
function Run_arm64() {
    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1'
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt
        if [ $? = 0 ]; then
            run_result='Run_arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='Run_arm64:'${model_name}' fail <<===========================this is the failed case'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1'
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt
        if [ $? = 0 ]; then
            run_result='Run_arm64:'${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='Run_arm64:'${model_name}' fail <<===========================this is the failed case'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_caffe_config}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "----------------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "Testcase Result                                                                                                                               "
    echo -e "-------- ------                                                                                                                               "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "----------------------------------------------------------------------------------------------------------------------------------------------"
}


basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_benchmark_nets.sh -a /home/temp_test -c /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408"
while getopts "a:c:m:d:" opt; do
    case ${opt} in
        a)
	    arm_path=${OPTARG}
            echo "arm_path is ${OPTARG}"
            ;;
        c)
	    convertor_path=${OPTARG}
            echo "convertor_path is ${OPTARG}"
            ;;
        m)
	    models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;		
        d)
	    device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# Unzip arm 
cd ${arm_path} || exit 1
tar -zxf MSLite-*-linux_arm64.tar.gz || exit 1

# Unzip convertor 
cd ${convertor_path} || exit 1
tar -zxf MSLite-*-linux_x86_64.tar.gz || exit 1
cd ${convertor_path}/MSLite-*-linux_x86_64 || exit 1
cp converter/converter_lite ./ || exit 1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib/:./third_party/protobuf/lib

# Convert the models
cd ${convertor_path}/MSLite-*-linux_x86_64 || exit 1

# Set models config filepath
models_tflite_config=${basepath}/models_tflite.cfg
models_caffe_config=${basepath}/models_caffe.cfg

rm -rf ${basepath}/ms_models
mkdir -p ${basepath}/ms_models
ms_models_path=${basepath}/ms_models

# Convert tflite models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name}
    echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}''
    ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}
done < ${models_tflite_config}

# Convert caffe models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name}
    pwd
    echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}''
    ./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}
done < ${models_caffe_config}

# Push to the arm and run benchmark:
# First:copy benchmark exe and so files to the server which connected to the phone
rm -rf ${basepath}/benchmark_test
mkdir -p ${basepath}/benchmark_test
benchmark_test_path=${basepath}/benchmark_test
cd ${benchmark_test_path} || exit 1
cp -a ${arm_path}/MSLite-*-linux_arm64/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
cp -a ${arm_path}/MSLite-*-linux_arm64/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

# Copy the MindSpore models:
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

# Second:adb push all needed files to the phone
adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/

# Third:run adb ,run session ,check the result:
echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt

adb -s ${device_id} shell < adb_cmd.txt

# Write resulte to temp file 
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo 'cases :' > ${run_benchmark_result_file}

# Run on x86
Run_x86 &
Run_x86_PID=$!
sleep 1

# Run on arm64
Run_arm64 & 
Run_arm64_PID=$!

wait ${Run_x86_PID}
Run_x86_status=$?

wait ${Run_arm64_PID}
Run_arm64_status=$?

# Print all results:
MS_PRINT_TESTCASE_START_MSG
while read line; do
    echo ${line}
done < ${run_benchmark_result_file}
MS_PRINT_TESTCASE_END_MSG

# Check all result and return value
if [[ ${Run_x86_status} = 0 ]] && [[ ${Run_arm64_status} = 0 ]];then
    echo "Run_x86 and Run_arm64 is ended"
    exit 0
else
    echo "run failed"
    exit 1
fi
