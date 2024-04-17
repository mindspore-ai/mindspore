#!/bin/bash

function check_device() {
    npu-smi info >./device.log
    line_idx=0
    target_line=7
    while read rows; do
        let line_idx++
        if [ ${line_idx} -eq ${target_line} ]; then
            device=$(echo ${rows} | awk '{print $3}')
            if [[ ${device} =~ "910B" ]]; then
                device="910B"
            fi
            if [[ ${device} =~ "910PremiumA" ]]; then
                device="910A"
            fi 
            echo "${device}"
        fi
    done <./device.log
}

function ConfigAscend() {
    echo "Start to copy Ascend local file"
    models_path=/home/workspace/mindspore_dataset/mslite/models/hiai
    echo "config file name: "
    ls ${basepath}/../${config_folder}/models_ascend_large_model_cloud_infer.cfg
    echo "========== config file list end =============="
    cp ${basepath}/../${config_folder}/models_ascend_large_model_cloud_infer.cfg ${infer_test_path} || exit 1
    cp ${basepath}/../config/large_model/* ${config_path} || exit 1
    cp ${basepath}/python/test_large_model_inference.py ${infer_test_path} || exit 1

    echo "Copy file success"
    # source ascend env
    export ASCEND_HOME=/usr/local/Ascend/latest
    ls /usr/local/Ascend/latest/bin/
    export PATH=${ASCEND_HOME}/compiler/ccec_compiler/bin:${PATH}
    source ${ASCEND_HOME}/aarch64-linux/bin/setenv.bash
}

function run_infer_parallel() {
    cd ${infer_test_path} || exit 1
    echo "Start running benchmark models"
    device=$(check_device)
    echo "Device is ${device}"

    # Prepare the config file list
    large_model_cloud_infer_cfg_list=${infer_test_path}/models_ascend_large_model_cloud_infer.cfg
    echo "large_model_cloud_infer_cfg_list: ${large_model_cloud_infer_cfg_list}"
    infer_config_path=${config_path}/config.json
    for cfg_file in ${large_model_cloud_infer_cfg_list[*]}; do
        while read line; do
            line_info=${line}
            echo "line_info: ${line_info}"
            if [[ $line_info == \#* || $line_info == "" ]]; then
                continue
            fi
            start_device_id=0
            # model_file_prefix   model_type  context parallel_num
            model_file_prefix=$(echo ${line_info} | awk -F ';' '{print $1}')
            echo "model_file_prefix: ${model_file_prefix}"
            model_type=$(echo ${line_info} | awk -F ';' '{print $2}')
            echo "model_type: ${model_type}"
            context=$(echo ${line_info} | awk -F ';' '{print $3}')
            echo "context: ${context}"
            parallel_num=$(echo ${line_info} | awk -F ';' '{print $4}')
            echo "parallel_num: ${parallel_num}"

            echo "predict ${model_type}"
            for ((i = 0; i < ${parallel_num}; i++)); do
                rank_id=$i
                LOG_FILE=${log_path}/${model_type}_${rank_id}.log
                prompt_model_path="${models_path}/${model_file_prefix}_full_${i}.mindir"
                decoder_model_path="${models_path}/${model_file_prefix}_inc_${i}.mindir"
                device_id=$((i + start_device_id))
                echo "python test_large_model_inference.py --device_id ${device_id} --rank_id ${rank_id} --prompt_model_path $prompt_model_path \
                    --decoder_model_path ${decoder_model_path} --config_path ${infer_config_path} --context ${context} \
                    --device ${device} --model_type ${model_type} --parallel_num ${parallel_num}"
                python test_large_model_inference.py \
                    --device_id ${device_id} \
                    --rank_id ${rank_id} \
                    --prompt_model_path ${prompt_model_path} \
                    --decoder_model_path ${decoder_model_path} \
                    --context ${context} \
                    --device ${device} \
                    --model_type ${model_type} \
                    --parallel_num ${parallel_num} \
                    --config_path ${infer_config_path} >${LOG_FILE} 2>&1 &
            done

            wait
            cat $LOG_FILE

            if grep -q "The avg time cost" $LOG_FILE; then
                echo "test_large_model_inference run success"
                exit 0
            else
                echo "test_large_model_inference run failed"
                exit 1
            fi

        done <${cfg_file}
    done
}

while getopts "r:m:e:l:" opt; do
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

basepath=$(pwd)
# clear working dir

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

infer_test_path=${basepath}/infer_test
echo "Ascend base path is ${infer_test_path}"
rm -rf ${infer_test_path}
mkdir -p ${infer_test_path}
config_path=${infer_test_path}/predict_config
mkdir -p ${config_path}
log_path=${infer_test_path}/logs
mkdir -p ${log_path}
ConfigAscend

cd $release_path
echo "installing mslite whl..."
python3 -m pip uninstall -y mindspore_lite || exit 1
python3 -m pip install *.whl --user
echo "install mslite success !"

run_infer_parallel
