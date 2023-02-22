#!/bin/bash
source ./scripts/base_functions.sh

# Run converter for tensorrt x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf ${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Prepare the config file list
    local tensorrt_cfg_file_list=("$models_tensorrt_config" "$models_nvgpu_posttraining_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${tensorrt_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $x86_fail_not_return
}

function Run_TensorRT_Mpirun() {
  source /etc/profile.tensorrt8.5.1
  cd ${x86_path}/tensorrt || exit 1
  tar -zxf ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
  tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
  cd ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64/ || exit 1
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
  cp tools/benchmark/benchmark ./ || exit 1

  echo "start mpirun models..."
  export ENABLE_NEW_API=true
  data_path=${models_path}'/input_output/'
  inpath=${data_path}'input'
  outpath=${data_path}'output'

  # dis_matmul_
  echo "start mpirun dis_matmul_ ..."
  model_name='dis_matmul_.mindir.ms'
  model_file=${ms_models_path}'/'${model_name}
  input_files=${inpath}'/dis_matmul_.mindir.ms.bin'
  output_file=${outpath}'/dis_matmul_.mindir.ms.out'
  echo 'mpirun -np 2 ./benchmark --modelFile='${model_file}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --device=GPU' >> "${run_tensorrt_mpirun_log_file}"
  mpirun -np 2 ./benchmark --modelFile=${model_file} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --device=GPU >> ${run_tensorrt_mpirun_log_file}
  if [ $? = 0 ]; then
      run_result='TensorRT_Server: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
  else
      run_result='TensorRT_Server: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
  fi


  # wide_and_deep_
#   echo "start mpirun wide_and_deep_ ..."
#   model_name='wide_and_deep_.mindir.ms'
#   model_file=${ms_models_path}'/'${model_name}
#   input_files=${inpath}'/wide_and_deep_.mindir.ms.bin_1,'${inpath}'/wide_and_deep_.mindir.ms.bin_2'
#   config_file=${inpath}'/wide_and_deep_.mindir.ms.config'
#   output_file=${outpath}'/wide_and_deep_.mindir.ms.out'
#   echo 'mpirun -np 2 ./benchmark --modelFile='${model_file}' --configFile='${config_file}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --device=GPU' >> "${run_tensorrt_mpirun_log_file}"
#   mpirun -np 2 ./benchmark --modelFile=${model_file} --inDataFile=${input_files} --configFile=${config_file} --benchmarkDataFile=${output_file} --device=GPU >> ${run_tensorrt_mpirun_log_file}
#   if [ $? = 0 ]; then
#       run_result='TensorRT_Server: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
#   else
#       run_result='TensorRT_Server: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
#   fi

  # wide_deep_worker_
#   echo "start mpirun wide_deep_worker_ ..."
#   export BENCHMARK_UPDATE_CONFIG_ENV=0
#   model_name='wide_deep_worker_.mindir.ms'
#   model_file=${ms_models_path}'/'${model_name}
#   input_files=${inpath}'/wide_deep_worker_.mindir.ms.bin_1,'${inpath}'/wide_deep_worker_.mindir.ms.bin_2'
#   output_file=${outpath}'/wide_deep_worker_.mindir.ms.out'
#   echo 'mpirun -np 2 ./benchmark --modelFile='${model_file}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --device=GPU' >> "${run_tensorrt_mpirun_log_file}"
#   mpirun -np 2 ./benchmark --modelFile=${model_file} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --device=GPU >> ${run_tensorrt_mpirun_log_file}
#   if [ $? = 0 ]; then
#       run_result='TensorRT_Server: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
#   else
#       run_result='TensorRT_Server: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
#   fi
#   export BENCHMARK_UPDATE_CONFIG_ENV=
}

# Run on NVIDIA TensorRT platform:
function Run_TensorRT() {
    source /etc/profile.tensorrt8.5.1
    cd ${x86_path}/tensorrt || exit 1
    tar -zxf ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/tensorrt/mindspore-lite-${version}-linux-x64/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
            mode model_file input_files output_file data_path acc_limit enableFp16 \
            run_result spec_cosine_limit
    # Prepare the config file list
    local tensorrt_cfg_file_list=("$models_tensorrt_config" "$models_nvgpu_posttraining_config")
    for cfg_file in ${tensorrt_cfg_file_list[*]}; do
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
            spec_cosine_limit=`echo ${accuracy_info} | awk -F ';' '{print $2}'`

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
            config_file_path=${data_path}'input/'${model_name}'.config'

            # set accuracy limitation
            acc_limit="0.5"
            if [[ ${spec_acc_limit} != "" ]]; then
                acc_limit="${spec_acc_limit}"
            elif [[ ${mode} == "fp16" ]]; then
                acc_limit="5"
            fi
            # set cosind distance limit
            cosine_limit="-1.1"
            if [[ ${spec_cosine_limit} != "" ]]; then
                cosine_limit="${spec_cosine_limit}"
            fi
            # whether enable fp16
            enableFp16="false"
            if [[ ${mode} == "fp16" ]]; then
                enableFp16="true"
            fi
            if [[ ${mode} == "offline_resize" ]]; then
                input_shapes=""
            fi

            # different tensorrt run mode use different cuda command
            echo 'CUDA_VISIBLE_DEVICES='${cuda_device_id}' ./benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --configFile='${config_file_path}' --cosineDistanceThreshold=${cosine_limit} --device=GPU' >> "${run_tensorrt_log_file}"
            CUDA_VISIBLE_DEVICES=${cuda_device_id} ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --configFile=${config_file_path} --cosineDistanceThreshold=${cosine_limit} --device=GPU >> ${run_tensorrt_log_file}
            CUDA_VISIBLE_DEVICES=${cuda_device_id} ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --configFile=${config_file_path} --cosineDistanceThreshold=${cosine_limit} --device=GPU >> ${run_tensorrt_log_file}

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

function Kill_Pid() {
    ## kill previous pid
    whoami=$(whoami)
    IFS=" " read -r -a pids <<< "$(ps -ef | grep benchmark | grep ${whoami} | awk -F ' ' '{print $2}')"
    for pid in ${pids[*]}; do
        echo "killing previous user pid ${pid} for ${whoami}"
        kill -9 ${pid}
    done
}

# Example:sh run_benchmark_tensorrt.sh -r /home/temp_test -m /home/temp_test/models -e x86_gpu -d 192.168.1.1:0
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
x86_fail_not_return="OFF" # This value could not be set to ON.
basepath=$(pwd)/${cuda_device_id}
rm -rf ${basepath}
mkdir -p ${basepath}
echo "NVIDIA TensorRT, bashpath is ${basepath}"

x86_path=${release_path}/centos_x86
file_name=$(ls ${x86_path}/*-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
ms_models_path=${basepath}/ms_models

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_tensorrt_config=${basepath}/../../${config_folder}/models_tensorrt.cfg
echo ${models_tensorrt_config}
models_nvgpu_posttraining_config=${basepath}/../../config/models_nvgpu_posttraining.cfg
echo ${models_nvgpu_posttraining_config}

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "Start run converter in tensorrt ..."
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
# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
fi

####################  run simple tensorrt models
run_tensorrt_log_file=${basepath}/run_tensorrt_log.txt
echo 'run tensorrt benchmark logs: ' > ${run_tensorrt_log_file}
run_benchmark_result_file=${basepath}/run_tensorrt_result_filess.txt
echo "Running in tensorrt on device ${cuda_device_id} ..."
Run_TensorRT &
Run_TensorRT_PID=$!
sleep 1

wait ${Run_TensorRT_PID}
Run_benchmark_status=$?

# Check converter result and return value
echo "Run x86 TensorRT GPU ended on device ${cuda_device_id}"

if [[ ${Run_benchmark_status} != 0 ]];then
    echo "Run_TensorRT failed"
    cat ${run_tensorrt_log_file}
    Print_Benchmark_Result $run_benchmark_result_file
    exit ${Run_benchmark_status}
fi

####################  run distribution tensorrt models
run_tensorrt_mpirun_log_file=${basepath}/run_tensorrt_mpirun_log.txt
echo 'run tensorrt mpirun logs: ' > ${run_tensorrt_mpirun_log_file}

echo "Running in tensorrt with mpirun"
export GLOG_v=1
Run_TensorRT_Mpirun &
Run_TensorRT_Mpirun_PID=$!
sleep 1

wait ${Run_TensorRT_Mpirun_PID}
Run_benchmark_status=$?

# Check converter result and return value
echo "Run x86 TensorRT GPU with mpirun ended"

if [[ ${Run_benchmark_status} != 0 ]];then
    echo "Run_TensorRT_Mpirun failed"
    cat ${run_tensorrt_mpirun_log_file}
fi

Print_Benchmark_Result $run_benchmark_result_file
exit 0
