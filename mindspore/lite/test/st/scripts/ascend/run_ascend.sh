#!/bin/bash
source ./scripts/base_functions.sh

function PrePareLocal() {
  echo "Start to copy local file"
  rm -rf ${benchmark_test_path}
  mkdir -p ${benchmark_test_path}

  cp ./scripts/base_functions.sh ${benchmark_test_path} || exit 1
  cp ./scripts/run_benchmark_python.sh ${benchmark_test_path} || exit 1
  cp -r ./python ${benchmark_test_path} || exit 1
  cp -r ./cpp ${benchmark_test_path} || exit 1
  cp -r ./java ${benchmark_test_path} || exit 1
  cp ./scripts/ascend/*.sh ${benchmark_test_path} || exit 1
  cp ./scripts/cloud_infer/run_benchmark_cloud_ascend.sh ${benchmark_test_path} || exit 1
  if [[ ${backend} =~ "_cloud" ]]; then
      models_ascend_config=./../${config_folder}/models_ascend_cloud.cfg
      if [[ ${backend} =~ "_ge" ]]; then
          models_ascend_config=./../${config_folder}/models_ascend_ge_cloud.cfg
      fi
      cp ${models_ascend_config} ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_python_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_mindir_cloud_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_mindir_cloud_java_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_with_config_cloud_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/ascend/*.config ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_on_the_fly_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_fake_model_full_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp -r ./../${config_folder}/quant ${benchmark_test_path} || exit 1
  else
      cp ./../${config_folder}/models_ascend_lite.cfg ${benchmark_test_path} || exit 1
  fi
  if [[ ${backend} =~ "arm" ]]; then
      if [[ ${backend} =~ "_cloud" ]]; then
          cp ${release_path}/linux_aarch64/cloud_fusion/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
          cp ${release_path}/linux_aarch64/cloud_fusion/*.whl ${benchmark_test_path} || exit 1
      else
          cp ${release_path}/linux_aarch64/ascend/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      fi
  else
      if [[ ${backend} =~ "_cloud" ]]; then
          cp ${release_path}/centos_x86/cloud_fusion/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      else
          cp ${release_path}/centos_x86/ascend/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      fi
  fi
  echo "Copy file success"
}

function PrePareRemote() {
  echo "Start to copy remote file"
  ssh ${user_name}@${device_ip} "rm -rf ${benchmark_test_path}; mkdir -p ${benchmark_test_path}" || exit 1

  scp ./scripts/run_benchmark_python.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp ./scripts/base_functions.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./python ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./cpp ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./java ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp ./scripts/ascend/*.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  if [[ ${backend} =~ "_cloud" ]]; then
      models_ascend_config=./../${config_folder}/models_ascend_cloud.cfg
      if [[ ${backend} =~ "_ge" ]]; then
          models_ascend_config=./../${config_folder}/models_ascend_ge_cloud.cfg
      fi
      scp ${models_ascend_config} ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/models_python_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_mindir_cloud_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_mindir_cloud_java_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_with_config_cloud_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/ascend/*.config ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  else
      scp ./../${config_folder}/models_ascend_lite.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  fi
  if [[ ${backend} =~ "arm" ]]; then
      if [[ ${backend} =~ "_cloud" ]]; then
          scp ${release_path}/linux_aarch64/cloud_fusion/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
          scp ${release_path}/linux_aarch64/cloud_fusion/*.whl ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      else
          scp ${release_path}/linux_aarch64/ascend/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      fi
  else
      if [[ ${backend} =~ "_cloud" ]]; then
          scp ${release_path}/centos_x86/cloud_fusion/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      else
          scp ${release_path}/centos_x86/ascend/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      fi
  fi
  echo "Copy file success"
}

function Run_Ascend() {
  if [ ${is_local} = 0 ]; then
    cd ${benchmark_test_path} || exit 1
    sh run_converter_ascend.sh ${backend} ${device_id} ${arch} ${ascend_fail_not_return}
  else
    ssh ${user_name}@${device_ip} "cd ${benchmark_test_path}; sh run_converter_ascend.sh ${backend} ${device_id} ${arch}"
  fi
  if [[ $? = 0 ]]; then
    run_result="run in ${backend} pass"; echo ${run_result} >> ${run_ascend_result_file};
  else
    run_result="run in ${backend} failed"; echo ${run_result} >> ${run_ascend_result_file}; exit 1
  fi
}

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -e Ascend310 -d 10.92.9.100:2
while getopts "r:m:d:e:l:p:" opt; do
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
            device_id=`echo ${OPTARG} | cut -d \: -f 2`
            echo "device_ip is ${device_ip}, ascend_device_id is ${device_id}."
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        p)
            ascend_fail_not_return=${OPTARG}
            echo "ascend_fail_not_return is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

if [[ ${backend} =~ "x86" ]]; then
  arch="x64"
elif [[ ${backend} =~ "arm" ]]; then
  arch="aarch64"
fi

config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

user_name=${USER}
echo "Current user name is ${user_name}"
basepath=$(pwd)/"${backend}_log_${device_id}"
rm -rf ${basepath}
mkdir -p ${basepath}
echo "Ascend base path is ${basepath}, device_ip: ${device_ip}, device_id: ${device_id}"
benchmark_test_path=/home/${user_name}/benchmark_test/${device_id}

ls /dev/davinci0
is_local=$?
if [ ${is_local} = 0 ]; then
  PrePareLocal
  if [ $? != 0 ]; then
    echo "Prepare local failed"
    exit 1
  fi
else
  PrePareRemote
  if [ $? != 0 ]; then
    echo "Prepare remote failed"
    exit 1
  fi
fi

# Write converter result to temp file
run_ascend_result_file=${basepath}'/run_'${backend}'_result.txt'
echo ' ' > ${run_ascend_result_file}

echo "Start to run in ${backend} ..."
Run_Ascend
Run_ascend_status=$?

run_converter_log_file=${basepath}'/run_'${backend}'_converter_log.txt'
run_converter_result_file=${basepath}'/run_'${backend}'_converter_result.txt'
run_benchmark_log_file=${basepath}'/run_'${backend}'_benchmark_log.txt'
run_benchmark_result_file=${basepath}'/run_'${backend}'_benchmark_result.txt'
if [ ${is_local} = 0 ]; then
  cp ${benchmark_test_path}/run_converter_log.txt ${run_converter_log_file} || exit 1
  cp ${benchmark_test_path}/run_converter_result.txt ${run_converter_result_file} || exit 1
  cp ${benchmark_test_path}/run_benchmark_log.txt ${run_benchmark_log_file} || exit 1
  cp ${benchmark_test_path}/run_benchmark_result.txt ${run_benchmark_result_file} || exit 1
else
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_converter_log.txt ${run_converter_log_file} || exit 1
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_converter_result.txt ${run_converter_result_file} || exit 1
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_log.txt ${run_benchmark_log_file} || exit 1
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_result.txt ${run_benchmark_result_file} || exit 1
fi
echo "Run in ${backend} ended"
cat ${run_ascend_result_file}
exit ${Run_ascend_status}
