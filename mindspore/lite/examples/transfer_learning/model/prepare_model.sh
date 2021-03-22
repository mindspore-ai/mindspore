#!/bin/bash

if [ ! -f "efficient_net_b0.ckpt" ]; then
  echo "Pretrained model weights are missing, downloading efficient_net_b0.ckpt"
  wget https://download.mindspore.cn/model_zoo/official/lite/efficient_net/efficient_net_b0.ckpt
fi

echo "============Exporting=========="
if [ -n "$1" ]; then
  DOCKER_IMG=$1
  rm -f *.so*
  docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python transfer_learning_export.py; chmod 444 transfer_learning_tod*.mindir; rm -rf __pycache__"
else
  echo "MindSpore docker was not provided, attempting to run locally"
  python transfer_learning_export.py
fi

CONVERTER="../../../build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
  if ! command -v converter_lite &> /dev/null
  then
    tar -xzf ../../../../../output/mindspore-lite-*-train-linux-x64.tar.gz --strip-components 4 --wildcards --no-anchored converter_lite libmindspore_gvar.so
    tar -xzf ../../../../../output/mindspore-lite-*-train-linux-x64.tar.gz --strip-components 6 --wildcards --no-anchored libglog.so.0
    if [ -f ./converter_lite ]; then
      CONVERTER=./converter_lite
    else
      echo "converter_lite could not be found in MindSpore build directory nor in system path"
      exit 1
    fi
  else
    CONVERTER=converter_lite
  fi
fi

echo "============Converting========="
pwd
LD_LIBRARY_PATH=./ $CONVERTER --fmk=MINDIR  --trainModel=false --modelFile=transfer_learning_tod_backbone.mindir --outputFile=transfer_learning_tod_backbone
LD_LIBRARY_PATH=./ $CONVERTER --fmk=MINDIR --trainModel=true --modelFile=transfer_learning_tod_head.mindir --outputFile=transfer_learning_tod_head
