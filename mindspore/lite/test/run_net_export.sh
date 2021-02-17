#!/bin/bash

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        printf "%-15s %-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo ${basepath}
# Set models default config filepath
models_mindspore_train_config=${basepath}/models_ms_train.cfg


# Example:run_net_export.sh -m /home/emir/Work/TestingEnv/train_models 
epoch_num=1
while getopts "c:m:t:" opt; do
    case ${opt} in
        c)
            models_mindspore_train_config=${OPTARG}
            echo "models_mindspore_train_config is ${models_mindspore_train_config}"
            ;;
        m)
            models_path=${OPTARG}"/models_train"
            echo "models_path is ${OPTARG}"
            ;;        
        t)
            epoch_num=${OPTARG}
            echo "train epoch num is ${OPTARG}"
            ;;                    
        ?)
            echo "unknown para"
            exit 1;;
    esac
done


logs_path=${basepath}/logs_train
rm -rf ${logs_path}
mkdir -p ${logs_path}

docker_image=mindspore/mindspore-gpu:1.1.0
# Export models
echo "Start Exporting models ..."
# Set log files
export_log_file=${logs_path}/export_log.txt
echo ' ' > ${export_log_file}

export_result_file=${logs_path}/export_result.txt
echo ' ' > ${export_result_file}

# Run export according to config file 
cd $models_path || exit 1
if [[ -z "${CLOUD_MODEL_ZOO}" ]]; then
    echo "CLOUD_MODEL_ZOO is not defined - exiting export models"
    exit 1
fi  

# Export mindspore train models:
while read line; do
    model_name=${line}
    if [[ $model_name == \#* ]]; then
        continue
    fi
    echo ${model_name}'_train_export.py' >> "${export_log_file}"
    echo 'exporting' ${model_name}
    echo 'docker run --user '"$(id -u):$(id -g)"' --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true '${docker_image}' python '${models_path}'/'${model_name}'_train_export.py' >>  "${export_log_file}"
    docker run --user "$(id -u):$(id -g)" --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true "${docker_image}" python ${models_path}'/'${model_name}_train_export.py "${epoch_num}"
    if [ $? = 0 ]; then
        export_result='export mindspore '${model_name}'_train_export pass';echo ${export_result} >> ${export_result_file}
    else
        export_result='export mindspore '${model_name}'_train_export failed';echo ${export_result} >> ${export_result_file}
    fi
done < ${models_mindspore_train_config}

Print_Result ${export_result_file}
 
