#!/bin/bash

echo "Begin run run_device_mem_test"
echo "cpp dir: ${LITE_ST_CPP_DIR}"
echo "model path: ${LITE_ST_MODEL}"
echo "lite home: ${LITE_HOME}"

source /usr/local/Ascend/latest/bin/setenv.bash
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH

cd ${LITE_ST_CPP_DIR}/device_example_cpp || exit 1

bash build.sh Ascend
if [ ! -f "./build/mindspore_quick_start_cpp" ];then
  echo "Failed to build device_example_cpp"
  exit 1
fi

build/mindspore_quick_start_cpp ${LITE_ST_MODEL}
Run_device_example_status=$?
if [[ ${Run_device_example_status} != 0 ]];then
  echo "Run device example failed"
else
  echo "Run device example success"
fi
exit ${Run_device_example_status}
