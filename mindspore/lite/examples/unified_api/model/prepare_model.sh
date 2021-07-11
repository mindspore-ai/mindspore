#!/bin/bash

if [[ -z ${EXPORT} ]]; then
  echo "============Exporting=========="
    rm -f lenet_tod.mindir
  if [ -n "$2" ]; then
    DOCKER_IMG=$2
    docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "PYTHONPATH=../../../../../model_zoo/official/cv/lenet/src python lenet_export.py '$1'; chmod 444 lenet_tod.mindir; rm -rf __pycache__"
  else
    echo "MindSpore docker was not provided, attempting to run locally"
    PYTHONPATH=../../../../../model_zoo/official/cv/lenet/src python lenet_export.py $1
  fi
fi


CONVERTER="../../../build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
  if ! command -v converter_lite &> /dev/null
  then
    tar -xzf ../../../../../output/mindspore-lite-*-linux-x64.tar.gz --strip-components 4 --wildcards --no-anchored converter_lite libglog.so.0 libmslite_converter_plugin.so
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
QUANT_OPTIONS=""
if [[ ! -z ${QUANTIZE} ]]; then
  echo "Quantizing weights"
  QUANT_OPTIONS="--quantType=WeightQuant --bitNum=8 --quantWeightSize=100 --quantWeightChannel=15"
fi
LD_LIBRARY_PATH=./ $CONVERTER --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod $QUANT_OPTIONS

