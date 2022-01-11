#!/bin/bash
source /root/miniconda3/bin/activate base
source /home/ascend/scripts/base_functions.sh

# Run converter for ascend x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf ${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Prepare the config file list
    local ascend_cfg_file_list=("$models_ascend_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile; $6:faile_not_return;
    Convert "${ascend_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $x86_fail_not_return
}

# source ascend env
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/fwkacllib/ccec_compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/fwkacllib/lib64:${ASCEND_HOME}/driver/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/opp
export TBE_IMPL_PATH=${ASCEND_HOME}/opp/op_impl/built-in/ai_core/tbe
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}

basepath=/home/ascend
x86_fail_not_return="OFF" # This value could not be set to ON.
x86_path=${basepath}/release
file_name=$(ls ${x86_path}/*-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

models_path=${basepath}/models
ms_models_path=${basepath}/ms_models
ms_scripts_path=${basepath}/scripts
models_ascend_config=${basepath}/config/models_ascend.cfg

# Write converter result to temp file
run_converter_log_file=${ms_scripts_path}/log/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${ms_scripts_path}/log/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

backend=$1

# Run converter
echo "Start to run converter in ${backend} ..."
Run_Converter
if [[ $? = 0 ]]; then
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit 1
fi

# Run Benchmark
source ${ms_scripts_path}/run_benchmark_ascend.sh -v $version -b $backend
Run_Benchmark_status=$?
exit ${Run_Benchmark_status}

