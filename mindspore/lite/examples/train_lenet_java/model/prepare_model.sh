#!/bin/bash

if [[ -z ${EXPORT} ]]; then
  echo "============Exporting=========="
  if [ -n "$1" ]; then
    DOCKER_IMG=$1
    docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "PYTHONPATH=../../../../../tests/perf_test python lenet_export.py; chmod 444 lenet_tod.mindir; rm -rf __pycache__"
  else
    echo "MindSpore docker was not provided, attempting to run locally"
    PYTHONPATH=../../../../../tests/perf_test python lenet_export.py
  fi
  
  if [ ! -f "$CONVERTER" ]; then
    echo "converter_lite could not be found in MindSpore build directory nor in system path"
    exit 1
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
$CONVERTER --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod $QUANT_OPTIONS

