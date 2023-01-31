#!/bin/bash
source ./scripts/base_functions.sh

# Run on x86 platform:
function Run_x86() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-*.tar.gz || exit 1
    # $1:framework;
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-*' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-*/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./tools/converter/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/glog
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/dnnl
    cp tools/benchmark/benchmark ./ || exit 1
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${x86_cfg_file_list[*]}" $models_path $models_path $run_x86_log_file $run_benchmark_result_file 'x86' 'CPU' '' $x86_fail_not_return
}

# Example:sh run_benchmark_x86_cloud_cpu.sh -r /home/temp_test -m /home/temp_test/models -e x86_cloud_mindir
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
x86_path=${release_path}/centos_x86/cloud_fusion
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

models_mindir_config=${basepath}/../${config_folder}/cloud_infer/models_mindir_cloud.cfg
# Prepare the config file list
x86_cfg_file_list=("$models_mindir_config")

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_x86_log_file=${basepath}/run_x86_cloud_log.txt
echo 'run x86 cloud logs: ' > ${run_x86_log_file}

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "x86_cloud_mindir" ]]; then
    # Run on x86 cloud
    echo "start Run x86 mindir cloud  $backend..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86_cloud_mindir" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?
    # Check benchmark result and return value
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86_mindir_cloud failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi

echo "Run_x86_mindir_cloud is ended"
Print_Benchmark_Result $run_benchmark_result_file

exit ${isFailed}
