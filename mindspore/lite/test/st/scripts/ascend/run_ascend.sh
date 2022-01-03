#!/bin/bash
source ./scripts/base_functions.sh

function Run_Ascend() {
  # copy converter files to scripts
  scp ./scripts/ascend/run_remote_ascend.sh root@${device_ip}:${device_scripts_path} || exit 1
    # copy benchmark files to sc
  scp ./scripts/ascend/run_benchmark_ascend.sh root@${device_ip}:${device_scripts_path} || exit 1

  ssh root@${device_ip} "cd ${device_scripts_path}; sh run_remote_ascend.sh ${backend}"
  if [[ $? = 0 ]]; then
    run_result="run in ${backend} pass"; echo ${run_result} >> ${run_ascend_result_file};
  else
    run_result="run in ${backend} failed"; echo ${run_result} >> ${run_ascend_result_file}; exit 1
  fi
}

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -e ascend -d 8.92.9.131
while getopts "r:m:d:e:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_ip=`echo ${OPTARG} | cut -d \: -f 1`
            echo "device_ip is ${device_ip}."
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

basepath=$(pwd)/result
rm -rf ${basepath}
mkdir -p ${basepath}
echo "Ascend base path is ${basepath}"

device_release_path=/home/ascend/release
device_config_path=/home/ascend/config
device_scripts_path=/home/ascend/scripts

ssh root@${device_ip} "sh /home/ascend/clear.sh" || exit 1
# copy release to device
scp ${release_path}/centos_x86/ascend/*-linux-x64.tar.gz root@${device_ip}:${device_release_path} || exit 1
# copy config to device
scp ./../config/models_ascend.cfg root@${device_ip}:${device_config_path} || exit 1
# copy comm func to device
scp ./scripts/base_functions.sh root@${device_ip}:${device_scripts_path} || exit 1

# Write converter result to temp file
run_ascend_result_file=${basepath}'/run_'${backend}'_result.txt'
echo ' ' > ${run_ascend_result_file}

echo "Start to run in ${backend} ..."
Run_Ascend
Run_ascend_status=$?

run_converter_log_file=${basepath}'/run_'${backend}'_converter_log.txt'
run_benchmark_log_file=${basepath}'/run_'${backend}'_benchmark_log.txt'
scp root@${device_ip}:${device_scripts_path}/log/run_converter_log.txt ${run_converter_log_file} || exit 1
scp root@${device_ip}:${device_scripts_path}/log/run_benchmark_log.txt ${run_benchmark_log_file} || exit 1

echo "Run in ${backend} ended"
Print_Converter_Result ${run_ascend_result_file}
exit ${Run_ascend_status}
