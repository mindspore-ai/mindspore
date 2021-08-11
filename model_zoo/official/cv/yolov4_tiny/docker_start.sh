#!/bin/bash

docker_image=$1 \
data_dir=$2 \
model_dir=$3 \

docker run -it --ipc=host \
              --device=/dev/davinci0 \
              --device=/dev/davinci1 \
              --device=/dev/davinci2 \
              --device=/dev/davinci3 \
              --device=/dev/davinci4 \
              --device=/dev/davinci5 \
              --device=/dev/davinci6 \
              --device=/dev/davinci7 \
              --device=/dev/davinci_manager \
              --device=/dev/devmm_svm --device=/dev/hisi_hdc \
              -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
              -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
              -v ${model_dir}:${model_dir} \
              -v ${data_dir}:${data_dir}  \
              -v /root/ascend/log:/root/ascend/log ${docker_image} /bin/bash
