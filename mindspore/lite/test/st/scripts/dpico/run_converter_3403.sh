#!/bin/bash

# Build x86 tar.gz file for dpico
function Run_Build_x86() {
  export MSLITE_REGISTRY_DEVICE=SD3403
  unset JAVA_HOME
  bash ${mindspore_top_dir}/build.sh -I x86_64 -j 80
  if [ $? = 0 ]; then
    echo "build x86 for dpico success"
    cp ${mindspore_top_dir}/output/*linux-x64.tar.gz ${x86_path}
    mkdir -p ${x86_path}/lib
    cp ${mindspore_top_dir}/mindspore/lite/build/_deps/opencv-4.2-for-dpico-src/lib/* ${x86_path}/lib
    cp ${mindspore_top_dir}/mindspore/lite/build/_deps/protobuf-3.9-for-dpico-src/lib/* ${x86_path}/lib
    cp ${mindspore_top_dir}/mindspore/lite/build/_deps/pico_mapper-src/lib/* ${x86_path}/lib
  else
    echo "build x86 for dpico failed"; return 1
  fi
}

# Build arm32 tar.gz file for dpico
function Run_Build_arm64() {
  export MSLITE_REGISTRY_DEVICE=SD3403
  unset JAVA_HOME
  bash ${mindspore_top_dir}/build.sh -I arm64 -j 80
  if [ $? = 0 ]; then
    echo "build arm64 for dpico success"
    cp ${mindspore_top_dir}/output/*linux-aarch64.tar.gz ${arm64_path}
  else
    echo "build arm64 for dpico failed"; return 1
  fi
}

function Run_Converter_CI_MODELS() {
  framework=$1
  if [[ ${framework} == 'TF' ]]; then
    model_location='tf'
  elif [[ ${framework} == 'ONNX' ]]; then
    model_location='onnx'
  else
    echo "unsupported framework"; return 1
  fi
  models_3403_cfg=$2
  while read line; do
      dpico_line_info=${line}
      if [[ $dpico_line_info == \#* || $dpico_line_info == "" ]]; then
        continue
      fi
      model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $1}'`
      model_name=${model_info%%;*}
      length=`expr ${#model_name} + 1`
      input_shape=${model_info:${length}}
      cfg_path_name=${models_path}/${model_location}/cfg_8bit/${model_name}.cfg
      cp ${cfg_path_name} ./ || exit 1
      ms_config_file=./converter_for_dpico.cfg
      echo '[registry]' > ${ms_config_file}
      echo 'plugin_path=./tools/converter/providers/SD3403/libdpico_atc_adapter.so' >> ${ms_config_file}
      echo -e 'disable_fusion=on\n' >> ${ms_config_file}
      echo '[dpico]' >> ${ms_config_file}
      echo 'dpico_config_path='./${model_name}.cfg >> ${ms_config_file}
      echo -e 'benchmark_path=./tools/benchmark/benchmark' >> ${ms_config_file}
      echo ${model_name} >> "${run_converter_log_file}"
      echo './converter_lite --inputDataFormat=NCHW --fmk='${framework}' --inputShape='${input_shape} '--modelFile='${models_path}'/'${model_location}'/models/'${model_name}' --configFile='${ms_config_file}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
      ./converter_lite --inputDataFormat=NCHW --inputShape=${input_shape} --fmk=${framework} --modelFile=${models_path}/${model_location}/models/${model_name} --configFile=${ms_config_file} --outputFile=${ms_models_path}/${model_name}
      if [ $? = 0 ]; then
          converter_result='converter '${framework}' '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
      else
          converter_result='converter '${framework}' '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}; exit 1
      fi
  done < ${models_3403_cfg}
}

# Run converter for DPICO models on x86 platform:
function Run_Converter() {
    cd ${x86_path} || exit 1
    tar -zxf mindspore-enterprise-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-enterprise-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./runtime/lib/:./tools/converter/third_party/glog/lib:./tools/converter/providers/SD3403/:${x86_path}/lib
    chmod +x ./tools/benchmark/benchmark

    echo ' ' > ${run_converter_log_file}
    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}
    chmod +x converter_lite
    # Convert dpico models:
    while read line; do
        dpico_line_info=${line}
        if [[ $dpico_line_info == \#* || $dpico_line_info == "" ]]; then
          continue
        fi
        model_location=`echo ${dpico_line_info}|awk -F ' ' '{print $1}'`
        model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $2}'`
        model_name=${model_info%%;*}
        echo ${model_name} >> "${run_converter_log_file}"
        # generate converter_lite config file
        cp ${models_path}/${model_location}/${model_name}.cfg ./ || exit 1
        ms_config_file=./converter_for_dpico.cfg
        echo '[registry]' > ${ms_config_file}
        echo 'plugin_path=./tools/converter/providers/SD3403/libdpico_atc_adapter.so' >> ${ms_config_file}
        echo -e 'disable_fusion=on\n' >> ${ms_config_file}
        echo '[dpico]' >> ${ms_config_file}
        echo 'dpico_config_path='./${model_name}.cfg >> ${ms_config_file}
        echo -e 'benchmark_path=./tools/benchmark/benchmark' >> ${ms_config_file}

        echo './converter_lite --inputDataFormat=NCHW --fmk=CAFFE --modelFile='${models_path}'/'${model_location}'/model/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_location}'/model/'${model_name}'.caffemodel --configFile='${ms_config_file}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite --inputDataFormat=NCHW --fmk=CAFFE --modelFile=${models_path}/${model_location}/model/${model_name}.prototxt --weightFile=${models_path}/${model_location}/model/${model_name}.caffemodel --configFile=${ms_config_file} --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter CAFFE '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter CAFFE '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_caffe_3403_config}

    Run_Converter_CI_MODELS 'TF' ${models_tf_3403_config}
    Run_convert_tf_status=$?
    if [[ ${Run_convert_tf_status} = 0 ]];then
        echo "Run convert tf success"
    else
        echo "Run convert tf failed"
        exit 1
    fi
    Run_Converter_CI_MODELS 'ONNX' ${models_onnx_3403_config}
    Run_convert_onnx_status=$?
    if [[ ${Run_convert_onnx_status} = 0 ]];then
        echo "Run convert onnx success"
    else
        echo "Run convert onnx failed"
        exit 1
    fi
}

# Run benchmark on 3403:
function Run_Benchmark() {
  if [[ "${CI_3403_USERNAME}" && "${CI_3403_PASSWORD}" ]]; then
    username=${CI_3403_USERNAME}
    password=${CI_3403_PASSWORD}
  else
    echo "ERROR: ENV CI_3403_USERNAME or CI_3403_PASSWORD not found."
    exit 1
  fi
  sshpass -p "${password}" ssh ${username}@${device_ip} "cd /mnt/dpico/gate/benchmark_test/${cur_timestamp}; sh run_benchmark_3403.sh"
  if [ $? = 0 ]; then
    run_result='benchmark_3403: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file};
  else
    run_result='benchmark_3403: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; exit 1
  fi
}

mindspore_top_dir=$(pwd)
echo ${mindspore_top_dir}
x86_path=${mindspore_top_dir}/x86_release/
rm -rf ${x86_path}
mkdir -p ${x86_path}
arm64_path=${mindspore_top_dir}/arm64_release/
rm -rf ${arm64_path}
mkdir -p ${arm64_path}
#set -e
st_dir=${mindspore_top_dir}/mindspore/lite/test/st

# Example:sh run_dpico_nets.sh r /home/temp_test -m /home/temp_test/models -e arm32_3403D -d 192.168.1.1
while getopts "m:d:e:l:" opt; do
    case ${opt} in
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_ip=${OPTARG}
            echo "device_ip is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
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

# build x86
echo "start building x86..."
Run_Build_x86 &
Run_build_x86_PID=$!
sleep 1

wait ${Run_build_x86_PID}
Run_build_x86_status=$?
if [[ ${Run_build_x86_status} = 0 ]];then
    echo "Run build x86 success"
else
    echo "Run build x86 failed"
    exit 1
fi

# build arm64
echo "start building arm64..."
Run_Build_arm64 &
Run_build_arm64_PID=$!
sleep 1

wait ${Run_build_arm64_PID}
Run_build_arm64_status=$?
if [[ ${Run_build_arm64_status} = 0 ]];then
    echo "Run build arm64 success"
else
    echo "Run build arm64 failed"
    exit 1
fi

# Set filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_caffe_3403_config=${st_dir}/../${config_folder}/models_caffe_3403.cfg
models_onnx_3403_config=${st_dir}/../${config_folder}/models_onnx_3403.cfg
models_tf_3403_config=${st_dir}/../${config_folder}/models_tf_3403.cfg
run_benchmark_script=${st_dir}/scripts/dpico/run_benchmark_3403.sh

# Set version
cd ${x86_path}
file_name=$(ls *-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set ms models output path
ms_models_path=${st_dir}/ms_models

# Write converter result to temp file
run_converter_log_file=${st_dir}/run_converter_log.txt
#rm ${run_converter_log_file}
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${st_dir}/run_converter_result.txt
#rm ${run_converter_result_file}
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter for dpico models..."
Run_Converter &
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter for dpico models success"
    Print_Converter_Result
else
    echo "Run converter for dpico models failed"
    cat ${run_converter_log_file}
    Print_Converter_Result
    exit 1
fi

# Write benchmark result to temp file
run_benchmark_result_file=${st_dir}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

# Copy the MindSpore models:
cur_timestamp=$((`date '+%s'`*1000+10#`date '+%N'`/1000000))
benchmark_test_path=/home/dpico/gate/benchmark_test/${cur_timestamp}
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
cp -a ${models_caffe_3403_config} ${benchmark_test_path} || exit 1
cp -a ${models_onnx_3403_config} ${benchmark_test_path} || exit 1
cp -a ${models_tf_3403_config} ${benchmark_test_path} || exit 1
cp -a ${run_benchmark_script} ${benchmark_test_path} || exit 1

#copy related so file to shared folder
cd ${arm64_path} || exit 1
tar -zxf mindspore-enterprise-lite-${version}-linux-aarch64.tar.gz || exit 1
cd ${arm64_path}/mindspore-enterprise-lite-${version}-linux-aarch64/ || exit 1
chmod +x ${mindspore_top_dir}/mindspore/lite/build/tools/benchmark/benchmark
cp -a ${mindspore_top_dir}/mindspore/lite/build/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1
cp -a ${arm64_path}/mindspore-enterprise-lite-${version}-linux-aarch64/providers/SD3403/libdpico_acl_adapter.so ${benchmark_test_path}/libdpico_acl_adapter.so || exit 1
cp -a ${arm64_path}/mindspore-enterprise-lite-${version}-linux-aarch64/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
cp -a ${mindspore_top_dir}/mindspore/lite/build/_deps/34xx_sdk-src/lib/*so* ${benchmark_test_path} || exit 1

if [[ $backend == "all" || $backend == "arm64_3403" ]]; then
    # Run on 34xx
    Run_Benchmark &
    Run_benchmark_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm64_3403" ]]; then
    wait ${Run_benchmark_PID}
    Run_benchmark_status=$?
    if [[ ${Run_benchmark_status} != 0 ]];then
        echo "Run_benchmark_3403 failed"
        isFailed=1
    else
        echo "Run_benchmark_3403 success"
        isFailed=0
    fi
    rm -rf ${benchmark_test_path} || exit 1
fi

if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
