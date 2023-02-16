#!/bin/bash
source ./scripts/base_functions.sh

# Run converter on x86 platform:
function Run_Converter() {
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

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
      exit 1
    fi
}

function Run_TensorRT() {
    source /etc/profile.tensorrt8.5.1
    cd ${tensorrt_path} || exit 1
    tar -zxf ${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${tensorrt_path}/mindspore-lite-${version}-linux-x64/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
            mode model_file input_files output_file data_path acc_limit enableFp16 \
            run_result
    # Prepare the config file list
    for cfg_file in ${models_server_inference_cfg_file_list[*]}; do
        cfg_file_name=${cfg_file##*/}
        while read line; do
            line_info=${line}
            if [[ $line_info == \#* || $line_info == "" ]]; then
                continue
            fi

            # model_info     accuracy_limit      run_mode
            model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
            accuracy_info=`echo ${line_info} | awk -F ' ' '{print $2}'`
            spec_acc_limit=`echo ${accuracy_info} | awk -F ';' '{print $1}'`

            # model_info detail
            model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
            input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
            input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
            mode=`echo ${model_info} | awk -F ';' '{print $5}'`
            input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
            if [[ ${model_name##*.} == "caffemodel" ]]; then
                model_name=${model_name%.*}
            elif [[ ${cfg_file_name} =~ "_posttraining" ]]; then
                model_name=${model_name}"_posttraining"
            fi

            # converter for distribution models
            if [[ ${spec_acc_limit} == "CONVERTER" ]]; then
                echo "Skip ${model_name} ......"
                continue
            fi

            echo "Benchmarking ${model_name} ......"
            model_file=${ms_models_path}'/'${model_name}'.ms'
            input_files=""
            output_file=""
            data_path=${models_path}'/input_output/'
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.ms.bin'
            else
                for i in $(seq 1 $input_num)
                do
                input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.ms.out'

            # set accuracy limitation
            acc_limit="0.5"
            if [[ ${spec_acc_limit} != "" ]]; then
                acc_limit="${spec_acc_limit}"
            elif [[ ${mode} == "fp16" ]]; then
                acc_limit="5"
            fi
            # whether enable fp16
            enableFp16="false"
            if [[ ${mode} == "fp16" ]]; then
                enableFp16="true"
            fi
            echo 'CUDA_VISIBLE_DEVICES='${cuda_device_id}' ./benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --enableParallelPredict=true --device=GPU'
            CUDA_VISIBLE_DEVICES=${cuda_device_id} ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --enableParallelPredict=true --device=GPU

            if [ $? = 0 ]; then
                run_result='TensorRT: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
            else
                run_result='TensorRT: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
            fi

        done < ${cfg_file}
    done
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                    Testcase                                                                                 Result   "
    echo -e "---                    --------                                                                                 ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-------------------------------------------------------------------------------------------------------------------------"
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

# Example:sh run_benchmark_gpu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:l:" opt; do
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
            device_ip=`echo ${OPTARG} | cut -d \: -f 1`
            cuda_device_id=`echo ${OPTARG} | cut -d \: -f 2`
            echo "device_ip is ${device_ip}, cuda_device_id is ${cuda_device_id}."
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
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

run_fail_not_return="OFF"
basepath=$(pwd)
echo "NVIDIA TensorRT, bashpath is ${basepath}"
x86_path=${release_path}/centos_x86  # ../release_pkg/lite
tensorrt_path=${x86_path}/server/tensorrt/cuda-11.1

file_name=""
if [[ $backend == "all" || $backend == "server_inference_x86_gpu" ]]; then
    cd ${x86_path} || exit 1
    file_name=$(ls *linux-x64.tar.gz)
fi

IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd ${basepath}
rm -rf ./*

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
cp ${basepath}/../${config_folder}/models_server_inference_tensorrt.cfg ./
models_server_inference_config=${basepath}/models_server_inference_tensorrt.cfg

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
export GLOG_logtostderr=0
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

# Copy the MindSpore models:
echo "Push files and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

backend=${backend:-"all"}
isFailed=0
if [[ $backend == "all" || $backend == "server_inference_x86_gpu" ]]; then
    echo "start Run ..."
    Run_TensorRT
    Run_x86_status=$?
fi

if [[ $backend == "all" || $backend == "server_inference_x86_gpu" ]]; then
    if [[ ${Run_x86_status} != 0 ]];then
        echo "run x86 server inference failed"
        isFailed=1
    fi
fi

Print_Benchmark_Result ${run_benchmark_result_file}
echo "run x86_gpu_server_inference is ended"
exit ${isFailed}
