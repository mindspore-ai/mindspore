#!/bin/bash

if [[ -z ${EXPORT} ]]; then
  echo "============Exporting=========="
  if [ -n "$1" ]; then
    DOCKER_IMG=$1
    docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "PYTHONPATH=../../../../../model_zoo/official/cv/lenet/src python lenet_export.py; chmod 444 lenet_tod.mindir; rm -rf __pycache__"
  else
    echo "MindSpore docker was not provided, attempting to run locally"
    PYTHONPATH=../../../../../model_zoo/official/cv/lenet/src python lenet_export.py
  fi
  
  if [ ! -f "$CONVERTER" ]; then
    echo "converter_lite could not be found in MindSpore build directory nor in system path"
    exit 1
  fi
fi

echo "============Converting========="
QUANT_OPTIONS=""
if [[ ! -z ${QUANTIZE} ]]; then
  echo "Quantizing weights"
  QUANT_OPTIONS="--quantType=WeightQuant --bitNum=8 --quantWeightSize=100 --quantWeightChannel=15"
fi
$CONVERTER --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod $QUANT_OPTIONS

