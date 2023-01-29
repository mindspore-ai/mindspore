#!/bin/bash
source ./scripts/base_functions.sh
source ./scripts/run_benchmark_python.sh

# Run on x86 java platform:
function Run_x86_java() {
    cd ${x86_path} || exit 1
    mkdir java || exit 1
    cp ${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz ./java/ || exit 1
    cd ./java || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    # compile benchmark
    cd mindspore-lite-${version}-linux-x64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./runtime/third_party/glog
    echo "javac -cp ${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d ."
    javac -cp ${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d .

    # Run tflite converted models:
    while read line; do
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_java_log_file}"
        echo "java -classpath .:${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.ms '${models_path}'/input_output/input/${model_name}.ms.bin '${models_path}'/input_output/output/${model_name}.ms.out 1" >> "${run_x86_java_log_file}"
        java -classpath .:${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.mindir ${models_path}/input_output/input/${model_name}.bin ${models_path}/input_output/output/${model_name}.out 1 "Runner"
        if [ $? = 0 ]; then
            run_result='x86_java: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_java: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            cat ${run_x86_java_log_file}
            cat ${run_benchmark_result_file}
            exit 1
        fi
    done < ${models_java_config}
}


# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-*.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-*/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${x86_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $x86_fail_not_return
}

# Run on x86 platform:
function Run_x86() {
    # $1:framework;
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-*' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-*/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${x86_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_log_file $run_benchmark_result_file 'x86' 'CPU' '' $x86_fail_not_return
}

# Example:sh run_benchmark_x86.sh -r /home/temp_test -m /home/temp_test/models -e arm_cpu
while getopts "r:m:e:p:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        p)
            x86_fail_not_return=${OPTARG}
            echo "x86_fail_not_return is ${OPTARG}"
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

basepath=$(pwd)
echo ${basepath}
if [[ $backend == "linux_arm64_tflite" ]]; then
  x86_path=${release_path}/linux_aarch64/
else
  x86_path=${release_path}/centos_x86/cloud_fusion
fi
cd ${x86_path}
file_name=$(ls *-linux-*.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

models_tf_config=${basepath}/../${config_folder}/models_tf_cloud.cfg
models_onnx_config=${basepath}/../${config_folder}/models_onnx_cloud.cfg
# Prepare the config file list
x86_cfg_file_list=()
if [[ $backend == "x86_cloud_tf" ]]; then
  x86_cfg_file_list=("$models_tf_config")
elif [[ $backend == "x86_cloud_onnx" ]]; then
  x86_cfg_file_list=("$models_onnx_config")
fi

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter &
Run_converter_PID=$!
sleep 2

wait ${Run_converter_PID}
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
if [[ $(Exist_File_In_Path ${ms_models_path} ".mindir") != "true" ]]; then
  echo "No mindir model found in ${ms_models_path}, please check if config file is empty!"
  exit 1
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_x86_log_file=${basepath}/run_x86_cloud_log.txt
echo 'run x86 cloud logs: ' > ${run_x86_log_file}

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "x86_cloud_onnx" || $backend == "x86_cloud_tf" ]]; then
    # Run on x86 cloud
    echo "start Run x86 cloud $backend..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86_cloud_onnx" || $backend == "x86_cloud_tf" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?
    # Check benchmark result and return value
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86_cloud failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi

echo "Run_x86_cloud is ended"
Print_Benchmark_Result $run_benchmark_result_file

# run python ST
if [[ $backend == "all" || $backend == "x86_cloud_onnx" ]]; then
  models_python_config=${basepath}/../config_level0/models_python_cpu.cfg
  models_python_cfg_file_list=("$models_python_config")
  Run_python_ST ${basepath} ${x86_path} ${ms_models_path} ${models_path} "${models_python_cfg_file_list[*]}" "CPU"
  Run_python_status=$?
  if [[ ${Run_python_status} != 0 ]];then
      echo "Run_python_status failed"
      isFailed=1
  fi
fi

# run Java ST
if [[ $backend == "all" || $backend == "x86_cloud_onnx" ]]; then
  run_x86_java_log_file=${basepath}/run_x86_java_log.txt
  echo 'run x86 java logs: ' > ${run_x86_java_log_file}
  models_java_config=${basepath}/../config_level0/models_java_cpu_cloud.cfg
  Run_x86_java
  Run_java_status=$?
  if [[ ${Run_java_status} != 0 ]];then
    cat $run_x86_java_log_file
    echo "Run_java_status failed"
    isFailed=1
  fi
fi


exit ${isFailed}
