#!/bin/bash

# Run on x86 platform:
function Run_x86() {
    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "{run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
	    run_result='x86: '${model_name}' pass'
	    echo ${run_result} >> ${run_benchmark_result_file}
	else
	    run_result='x86: '${model_name}' failed'
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
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "${run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_caffe_config}

    # Run onnx converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "${run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_onnx_config}

    # Run tflite post training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "${run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}_posttraining.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --warmUpLoopCount=1 --loopCount=1 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_posttraining pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_posttraining failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "${run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1 --numThreads=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1 --numThreads=1 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_awaretraining pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_awaretraining failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Run mindspore converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  '${convertor_path}'/mindspore-lite-'${version}'-runtime-x86-'${process_unit_x86} >> "${run_benchmark_log_file}"
        cd ${convertor_path}/mindspore-lite-${version}-runtime-x86-${process_unit_x86} || return 1
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath='${ms_models_path}'/'${model_name}'.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib;./benchmark/benchmark --modelPath=${ms_models_path}/${model_name}.ms --inDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --calibDataPath=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --warmUpLoopCount=1 --loopCount=1 --accuracyThreshold=1.5 >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
    done < ${models_mindspore_config}
}

# Run on arm64 platform:
function Run_arm64() {
    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
        # run benchmark test without clib data
        #echo ${model_name}
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
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
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_caffe_config}

    # Run onnx converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "{run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_onnx_config}

    # Run fp16 converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1 --fp16Priority=true --accuracyThreshold=5' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1 --fp16Priority=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --fp16Priority=true --accuracyThreshold=5' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --fp16Priority=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
	    echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_fp16_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1 --fp16Priority=true --accuracyThreshold=5' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1 --fp16Priority=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_benchmark_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --fp16Priority=true --accuracyThreshold=5' >> "${run_benchmark_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --fp16Priority=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_benchmark_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'
            echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'
            echo ${run_result} >> ${run_benchmark_result_file}
            return 1
        fi
	#sleep 1
    done < ${models_tflite_awaretraining_config}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env        Testcase                                                                                                       Result   "
    echo -e "---        --------                                                                                                       ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
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

file_name=$(ls ${arm_path}/*runtime-arm64*.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
IFS="." read -r -a suffix <<< "${file_name_array[-1]}"
process_unit_arm=${suffix[0]}

file_name=$(ls ${convertor_path}/*runtime-x86*.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
IFS="." read -r -a suffix <<< "${file_name_array[-1]}"
process_unit_x86=${suffix[0]}

# Unzip arm
cd ${arm_path} || exit 1
tar -zxf mindspore-lite-${version}-runtime-arm64-${process_unit_arm}.tar.gz || exit 1

# Unzip x86 runtime and convertor
cd ${convertor_path} || exit 1
tar -zxf mindspore-lite-${version}-runtime-x86-${process_unit_x86}.tar.gz || exit 1

tar -zxf mindspore-lite-${version}-convert-ubuntu.tar.gz || exit 1
cd ${convertor_path}/mindspore-lite-${version}-convert-ubuntu || exit 1
cp converter/converter_lite ./ || exit 1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib/:./third_party/protobuf/lib

# Convert the models
cd ${convertor_path}/mindspore-lite-${version}-convert-ubuntu || exit 1

# Write resulte to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_benchmark_log_file=${basepath}/run_benchmark_log.txt
echo 'run benchmark logs: ' > ${run_benchmark_log_file}

# Set models config filepath
models_tflite_config=${basepath}/models_tflite.cfg
models_caffe_config=${basepath}/models_caffe.cfg
models_tflite_awaretraining_config=${basepath}/models_tflite_awaretraining.cfg
models_tflite_posttraining_config=${basepath}/models_tflite_posttraining.cfg
models_onnx_config=${basepath}/models_onnx.cfg
models_fp16_config=${basepath}/models_fp16.cfg
models_mindspore_config=${basepath}/models_mindspore.cfg
Convert_status=0

rm -rf ${basepath}/ms_models
mkdir -p ${basepath}/ms_models
ms_models_path=${basepath}/ms_models

# Convert tflite models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name} || Convert_status=$?
done < ${models_tflite_config}

# Convert caffe models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    #pwd >> ${run_benchmark_log_file}
    echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name} || Convert_status=$?
done < ${models_caffe_config}

# Convert onnx models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    #pwd >> ${run_benchmark_log_file}
    echo './converter_lite  --fmk=ONNX --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=ONNX --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name} || Convert_status=$?
done < ${models_onnx_config}

# Convert mindspore models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    pwd >> "${run_benchmark_log_file}"
    echo './converter_lite  --fmk=MS --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=MS --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name} || Convert_status=$?
done < ${models_mindspore_config}

# Convert TFLite PostTraining models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_posttraining' --quantType=PostTraining --config_file='${models_path}'/'${model_name}'_posttraining.config' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_posttraining --quantType=PostTraining --config_file=${models_path}/${model_name}_posttraining.config || Convert_status=$?
done < ${models_tflite_posttraining_config}

# Convert TFLite AwareTraining models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
      continue
    fi
    echo ${model_name} >> "${run_benchmark_log_file}"
    echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}' --quantType=AwareTraining' >> "${run_benchmark_log_file}"
    ./converter_lite  --fmk=TFLITE --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name} --quantType=AwareTraining || Convert_status=$?
done < ${models_tflite_awaretraining_config}

# Copy fp16 ms models:
while read line; do
  model_name=${line%.*}
  if [[ $model_name == \#* ]]; then
      continue
  fi
  echo 'cp '${ms_models_path}'/'${model_name}'.ms' ${ms_models_path}'/'${model_name}'.fp16.ms'
  cp ${ms_models_path}/${model_name}.ms ${ms_models_path}/${model_name}.fp16.ms
done < ${models_fp16_config}

# Check all result and return value
if [[ ${Convert_status} = 0 ]];then
    echo "convert is ended"
else
    echo "convert failed"
    cat ${run_benchmark_log_file}
    exit 1
fi

# Push to the arm and run benchmark:
# First:copy benchmark exe and so files to the server which connected to the phone
echo "Push files to the arm and run benchmark"
rm -rf ${basepath}/benchmark_test
mkdir -p ${basepath}/benchmark_test
benchmark_test_path=${basepath}/benchmark_test
cd ${benchmark_test_path} || exit 1
cp -a ${arm_path}/mindspore-lite-${version}-runtime-arm64-${process_unit_arm}/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
cp -a ${arm_path}/mindspore-lite-${version}-runtime-arm64-${process_unit_arm}/lib/liboptimize.so ${benchmark_test_path}/liboptimize.so || exit 1
cp -a ${arm_path}/mindspore-lite-${version}-runtime-arm64-${process_unit_arm}/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

# Copy the MindSpore models:
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

# Second:adb push all needed files to the phone
adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

# Third:run adb ,run session ,check the result:
echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt

adb -s ${device_id} shell < adb_cmd.txt

# Run on x86
echo "start Run x86 ..."
Run_x86 &
Run_x86_PID=$!
sleep 1

# Run on arm64
echo "start Run arm64 ..."
Run_arm64 &
Run_arm64_PID=$!

wait ${Run_x86_PID}
Run_x86_status=$?

wait ${Run_arm64_PID}
Run_arm64_status=$?

# Print all results:
MS_PRINT_TESTCASE_START_MSG
while read line; do
    arr=("${line}")
    printf "%-10s %-110s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
done < ${run_benchmark_result_file}
MS_PRINT_TESTCASE_END_MSG

# Check all result and return value
if [[ ${Run_x86_status} = 0 ]] && [[ ${Run_arm64_status} = 0 ]];then
    echo "Run_x86 and Run_arm64 is ended"
    exit 0
else
    echo "run failed"
    cat ${run_benchmark_log_file}
    
    #print the result table again:
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-10s %-110s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < ${run_benchmark_result_file}
    MS_PRINT_TESTCASE_END_MSG
    
    exit 1
fi
