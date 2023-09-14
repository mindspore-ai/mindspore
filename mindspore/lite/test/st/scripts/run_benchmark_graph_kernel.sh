#!/bin/bash
source ./scripts/base_functions.sh

# Run converter with graph_kernel
function Run_Converter() {
    echo "Start converting models"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch} || exit 1
    echo "current dir is:$(pwd)"

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib
    if [[  ${backend} =~ "ascend"  ]]; then
     source ${benchmark_test_path}/ascend_custom_op/mslite_tbe_and_aicpu/bin/set_env.bash
    fi
    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}
    echo "convert model cfg: "${models_server_inference_cfg_file_list[*]}
    for cfg_file in ${models_server_inference_cfg_file_list[*]}; do
        if [ ! -f ${cfg_file} ]; then
            echo "can not find model config file: ${cfg_file}"
            return 1
        fi
    done
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${models_server_inference_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $run_fail_not_return
    return $?
}

# Run on CPU,TensorRT or ACL
function Run_Benchmark() {
    echo "Start running benchmark models"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./tools/converter/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/glog
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/dnnl
    if [[  ${backend} =~ "ascend"  ]]; then
        source ${benchmark_test_path}/ascend_custom_op/mslite_tbe_and_aicpu/bin/set_env.bash
    fi
    cp tools/benchmark/benchmark ./ || exit 1

    echo "benchmark model cfg list:"
    echo ${models_server_inference_cfg_file_list[*]}

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
        mode model_file input_files output_file data_path acc_limit enableFp16 \
        run_result
    # Prepare the config file list
    for cfg_file in ${models_server_inference_cfg_file_list[*]}; do
        while read line; do
            line_info=${line}
            if [[ $line_info == \#* || $line_info == "" ]]; then
                continue
            fi

            # model_info     accuracy_limit      run_mode
            model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
            accuracy_info=$(echo ${line_info} | awk -F ' ' '{print $2}')
            spec_acc_limit=$(echo ${accuracy_info} | awk -F ';' '{print $1}')

            # model_info detail
            model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
            input_info=$(echo ${model_info} | awk -F ';' '{print $2}')
            input_shapes=$(echo ${model_info} | awk -F ';' '{print $3}')
            mode=$(echo ${model_info} | awk -F ';' '{print $5}')
            input_num=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}')
            if [[ ${model_name##*.} == "caffemodel" ]]; then
                model_name=${model_name%.*}
            fi

            # converter for distribution models
            if [[ ${spec_acc_limit} == "CONVERTER" ]]; then
                echo "Skip ${model_name} ......"
                continue
            fi
            use_parallel_predict="false"
            if [[ ${mode} =~ "parallel_predict" ]]; then
                use_parallel_predict="true"
            fi
            echo "Benchmarking ${model_name} ......"
            model_file=${ms_models_path}'/'${model_name}'.mindir'
            input_files=""
            output_file=""
            data_path=${models_path}'/input_output/'
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.bin'
            else
                for i in $(seq 1 $input_num); do
                    input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.out'

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
            benchmark_command="./benchmark --enableParallelPredict=${use_parallel_predict} --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=${benchmark_device} --configFile=${benchmark_config_file}"
            echo "${benchmark_command}"
            ${benchmark_command} >> ${run_benchmark_log_file}
            if [ $? = 0 ]; then
                if [[ ${mode} =~ "parallel_predict" ]]; then
                    run_result="${benchmark_device}: ${model_name} parallel_pass"
                    echo ${run_result} >>${run_benchmark_result_file}
                else
                    run_result="${benchmark_device}: ${model_name} pass"
                    echo ${run_result} >>${run_benchmark_result_file}
                fi
            else
                if [[ ${mode} =~ "parallel_predict" ]]; then
                    run_result="${benchmark_device}: ${model_name} parallel_failed"
                    echo ${run_result} >>${run_benchmark_result_file}
                    return 1
                else
                    run_result="${benchmark_device}: ${model_name} failed"
                    echo ${run_result} >>${run_benchmark_result_file}
                    return 1
                fi
            fi

        done <${cfg_file}
    done
}

function Run_Benchmark_GE() {
    echo "Start running benchmark ge backend"
    export ASCEND_BACK_POLICY="ge"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ || exit 1
    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}
    models_server_inference_cfg_file_list=$ge_cfg_file_list
    for cfg_file in ${models_server_inference_cfg_file_list[*]}; do
        while read line; do
            if [[ $line == \#* || $line == "" ]]; then
                continue
            fi
            model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
            if [[ ! $model_name =~ \.mindir ]]; then
                #skip models whose format is not mindir
                continue
            fi
            cp ${models_path}"/"${model_name} $ms_models_path/${model_name}".mindir" || exit 1
        done <${cfg_file}
    done
    # Empty config file is allowed, but warning message will be shown
    if [[ $(Exist_File_In_Path ${ms_models_path} ".mindir") != "true" ]]; then
        echo "No ms model found in ${ms_models_path}, please check if ge config files are empty!"
        exit 0
    fi
    # add config file path for ge
    benchmark_config_file=${benchmark_test_path}"/graph_kernel/graph_kernel_ascend.cfg"
    Run_Benchmark
    return $?
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done <$1
    MS_PRINT_TESTCASE_END_MSG
}

function Kill_Pid() {
    ## kill previous pid
    whoami=$(whoami)
    IFS=" " read -r -a pids <<<"$(ps -ef | grep benchmark | grep ${whoami} | awk -F ' ' '{print $2}')"
    for pid in ${pids[*]}; do
        echo "killing previous user pid ${pid} for ${whoami}"
        kill -9 ${pid}
    done
}

function ConfigCPU() {
    if [[ $backend =~ "arm" ]]; then
        release_package_path=${release_path}/linux_aarch64/cloud_fusion/
    else
        release_package_path=${release_path}/centos_x86/cloud_fusion/
    fi
    benchmark_device="CPU"
    models_server_inference_cfg_file_list=${basepath}/../${config_folder}/models_graph_kernel_cpu_cloud.cfg
}

function ConfigGPU() {
    source /etc/profile.tensorrt8.5.1
    echo "NVIDIA TensorRT, basepath is ${basepath}"
    release_package_path=${release_path}/centos_x86/cloud_fusion/ # ../release_pkg/lite
    models_server_inference_cfg_file_list=${basepath}/../${config_folder}/models_graph_kernel_gpu_cloud.cfg
    benchmark_device="GPU"
    export CUDA_VISIBLE_DEVICES=${device_id}
}

function InstallAscendCustomOps() {
    echo "prepare to install ascend custom op at: ${benchmark_test_path}/ascend_custom_op"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/
    install_script="bash tools/custom_kernels/ascend/tbe_and_aicpu/install.sh --install-path=${benchmark_test_path}/ascend_custom_op"
    ${install_script}
    install_result=$?
    if [ ${install_result} != 0 ]; then
        echo "install ascend custom op failed, run '${install_script}' failed."
        exit 1
    fi
    source ${benchmark_test_path}/ascend_custom_op/mslite_tbe_and_aicpu/bin/set_env.bash
    echo "Successfully installed ascend custom op, ASCEND_CUSTOM_OPP_PATH is: $ASCEND_CUSTOM_OPP_PATH"
}

function ConfigAscend() {
    echo "Start to copy Ascend local file"
    benchmark_device=Ascend
    user_name=${USER}
    echo "Current user name is ${user_name}"
    benchmark_test_path=/home/${user_name}/benchmark_test/${device_id}
    echo "Ascend base path is ${benchmark_test_path}, device_id: ${device_id}"
    rm -rf ${benchmark_test_path}
    mkdir -p ${benchmark_test_path}
    models_path=/home/workspace/mindspore_dataset/mslite/models/hiai
    mkdir -p ${benchmark_test_path}/graph_kernel
    cp ${basepath}/../${config_folder}/graph_kernel/graph_kernel_ascend.cfg ${benchmark_test_path}/graph_kernel/ || exit 1
    cp ${basepath}/../${config_folder}/models_graph_kernel_ascend_acl_cloud.cfg ${benchmark_test_path} || exit 1
    cp ${basepath}/../${config_folder}/models_graph_kernel_ascend_ge_cloud.cfg ${benchmark_test_path} || exit 1
    # we do not convert ge models, because we will use benchmark to run mindir with ge backend
    models_server_inference_cfg_file_list=${benchmark_test_path}/models_graph_kernel_ascend_acl_cloud.cfg
    ge_cfg_file_list=${benchmark_test_path}/models_graph_kernel_ascend_ge_cloud.cfg
    if [[ ${arch} = "aarch64" ]]; then
        release_package_path=${release_path}/linux_aarch64/cloud_fusion/ || exit 1
    else
        release_package_path=${release_path}/centos_x86/cloud_fusion/ || exit 1
    fi
    echo "Copy file success"
    # source ascend env
    export ASCEND_SLOG_PRING_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_HOME=/usr/local/Ascend/latest
    export PATH=${ASCEND_HOME}/compiler/ccec_compiler/bin:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${ASCEND_HOME}/../driver/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/opp
    export TBE_IMPL_PATH=${ASCEND_HOME}/opp/built-in/op_impl/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
}

# Example:sh run_benchmark_graph_kernel.sh -r /home/temp_test -m /home/temp_test/models -e x86_gpu -d 192.168.1.1:0
# backend can be: x86_gpu,x86_cpu,arm64_cpu,arm64_android_cpu,x86_ascend,arm64_ascend
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
        device_ip=$(echo ${OPTARG} | cut -d \: -f 1)
        device_id=$(echo ${OPTARG} | cut -d \: -f 2)
        echo "device_ip is ${device_ip}, device_id is ${device_id}."
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
        exit 1
        ;;
    esac
done

run_fail_not_return="OFF"
basepath=$(pwd)

# default working dir is benchmark_test_path
benchmark_test_path=${basepath}/benchmark_test

# clear working dir
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}

if [[ $backend =~ "arm" ]]; then
    arch="aarch64"
else
    arch="x64"
fi

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

if [[ ${backend} =~ "cpu" ]]; then
    ConfigCPU
elif [[ ${backend} =~ "gpu" ]]; then
    ConfigGPU
elif [[ ${backend} =~ "ascend" ]]; then
    ConfigAscend
else
    echo "Unknown backend for graph kernel, backend is: ${backend}"
    exit 1
fi

# get release package path and version
ms_models_path=${benchmark_test_path}/ms_models
cd $release_package_path
release_file=$(ls *-linux-*.tar.gz)
release_file_path="$release_package_path/$release_file"
IFS="-" read -r -a file_name_array <<<"$release_file"
version=${file_name_array[2]}
echo "Running MSLite with Graph Kernel on ${backend}, device is $benchmark_device , release file path is $release_file_path, working dir is: $benchmark_test_path"
cd -
# uncompressing package file
echo "uncompressing package file..."
cd ${benchmark_test_path} || exit 1
tar -zxf $release_file_path  || exit 1
echo "installing akg..."
python3 -m pip uninstall -y mindspore_lite || exit 1
python3 -m pip uninstall -y akg || exit 1
python3 -m pip install ./mindspore-lite-${version}-linux-${arch}/tools/akg/*.whl --user || exit 1

# install ascend custom op
if [[ ${backend} =~ "ascend" ]]; then
    InstallAscendCustomOps
    echo "------------------ Aacend env ------------------"
    env | grep "ASCEND"
    echo "------------------------------------------------"
fi

# Write converter result to temp file
run_converter_log_file=${benchmark_test_path}/run_converter_log.txt
echo ' ' >${run_converter_log_file}

run_converter_result_file=${benchmark_test_path}/run_converter_result.txt
echo ' ' >${run_converter_result_file}

# Run converter
echo "Start run converter with Graph Kernel Fusion ..."
Run_Converter
Run_converter_status=$?

# Check converter result and return value
if [[ ${Run_converter_status}  != 0 ]]; then
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit $Run_converter_status
else
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
fi

# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".mindir") != "true" ]]; then
    echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
    exit 0
fi

####################  run models
# Run converter
echo "Start run benchmark with Graph Kernel Fusion ..."
run_benchmark_log_file=${benchmark_test_path}/run_benchmark_log.txt
echo "run ${benchmark_device} benchmark logs: " > ${run_benchmark_log_file}
run_benchmark_result_file=${benchmark_test_path}/run_graph_kernel_result_files.txt
echo "Running in ${benchmark_device} on device ${device_id} ..."
Run_Benchmark
Run_benchmark_status=$?
if [[ ${Run_benchmark_status} != 0 ]]; then
    echo "Run_Benchmark failed"
    cat ${run_benchmark_log_file}
    Print_Benchmark_Result $run_benchmark_result_file
    exit ${Run_benchmark_status}
fi

# run ge backend
if [[ ${backend} =~ "ascend" ]]; then
    Run_Benchmark_GE &
    Run_Benchmark_GE_PID=$!

    wait ${Run_Benchmark_GE_PID}
    Run_benchmark_ge_status=$?

    if [[ ${Run_benchmark_ge_status} != 0 ]]; then
        echo "Run_Benchmark_GE failed"
        cat ${run_benchmark_log_file}
        Print_Benchmark_Result $run_benchmark_result_file
        exit ${Run_benchmark_ge_status}
    fi
fi

echo "Run_Benchmark success"
Print_Benchmark_Result $run_benchmark_result_file
exit 0
