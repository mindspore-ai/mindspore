#!/bin/bash

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_nnie_nets.sh r /home/temp_test -m /home/temp_test/models
while getopts "r:m:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
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

pkg_path=${release_path}/linux_aarch64/cloud_fusion/python37/triton

cd ${pkg_path} || exit 1
file_name=$(ls *linux-aarch64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
tar -zxf mindspore-lite-${version}-linux-aarch64.tar.gz || exit 1
export LD_LIBRARY_PATH=mindspore-lite-${version}-linux-aarch64/tools/converter/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=mindspore-lite-${version}-linux-aarch64/runtime/lib:${LD_LIBRARY_PATH}
cd - || exit 1

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_triton_config=${basepath}/../${config_folder}/models_triton.cfg

backend_directory=${pkg_path}/mindspore-lite-${version}-linux-aarch64/tools/providers/triton/
# Set ms models output path
model_repository=${models_path}/triton_models/

echo "Begin to start tritonserver."
tritonserver --model-repository=${model_repository} --backend-directory=${backend_directory} &
sleep 5 # wait to start the tritonserver.
echo "Begin to test client."
while read line; do
    line_info=${line}
    if [[ $line_info == \#* || $line_info == "" ]]; then
        continue
    fi
    model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
    model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
    input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
    input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
    output_names=`echo ${model_info} | awk -F ';' '{print $4}'`
    input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`

    python mslite_client.py -m $model_name --input_name $input_names --input_shape $input_shapes \
    --input_data ${model_repository}/${model_name}/input.bin \
    --output_name ${output_names} --output_data ${model_repository}/${model_name}/output.bin
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "Test tritonserver inference failed."
        pid=`pidof tritonserver`
        kill -9 $pid
        exit $status
    fi
done < $models_triton_config
echo "Test tritonserver inference success."
pid=`pidof tritonserver`
kill -9 $pid
exit 0
