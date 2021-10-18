#!/bin/bash

if [[ -z ${EXPORT} ]]; then
  echo "============Exporting=========="
    rm -f lenet_tod.mindir
  if [ -n "$2" ]; then
    DOCKER_IMG=$2
    docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "PYTHONPATH=../../../../../tests/perf_test python lenet_export.py '$1'; chmod 444 lenet_tod.mindir; rm -rf __pycache__"
  else
    echo "MindSpore docker was not provided, attempting to run locally"
    PYTHONPATH=../../../../../tests/perf_test python lenet_export.py $1
  fi
fi


CONVERTER="../../../build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
  if ! command -v converter_lite &> /dev/null
  then
    tar -xzf ../../../../../output/mindspore-lite-*-linux-x64.tar.gz --strip-components 4 --wildcards --no-anchored converter_lite *so.* *.so
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

function GenerateWeightQuantConfig() {
  echo "[common_quant_param]" > $4
  echo "quant_type=WEIGHT_QUANT" >> $4
  echo "bit_num=$1" >> $4
  echo "min_quant_weight_size=$2" >> $4
  echo "min_quant_weight_channel=$3" >> $4
}

echo "============Converting========="
QUANT_OPTIONS=""
if [[ ! -z ${QUANTIZE} ]]; then
  echo "Quantizing weights"
  WEIGHT_QUANT_CONFIG=ci_lenet_tod_weight_quant.cfg
  GenerateWeightQuantConfig 8 100 15 ${WEIGHT_QUANT_CONFIG}
  QUANT_OPTIONS="--configFile=${WEIGHT_QUANT_CONFIG}"
fi
LD_LIBRARY_PATH=./:${LD_LIBRARY_PATH} $CONVERTER --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod $QUANT_OPTIONS

