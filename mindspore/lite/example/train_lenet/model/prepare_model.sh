CONVERTER="../../../../../mindspore/lite/build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
  if ! command -v converter_lite &> /dev/null
  then
    echo "converter_lite could not be found in MindSpore build directory nor in system path"
    exit
  else
    CONVERTER=converter_lite
  fi
fi

echo "============Exporting=========="
if [ -n "$1" ]; then
  DOCKER_IMG=$1
  docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python lenet_export.py; chmod 444 lenet_tod.mindir; rm -rf __pycache__"
else
  echo "MindSpore docker was not provided, attempting to run locally"
  python lenet_export.py
fi


echo "============Converting========="
$CONVERTER --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod

