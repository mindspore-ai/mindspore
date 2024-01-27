#!/bin/bash

function SetUpAscendEnv() {
    export ASCEND_HOME_PATH=/usr/local/Ascend/latest
    ls /usr/local/Ascend/
    . ${ASCEND_HOME_PATH}/bin/setenv.bash
    export ASCEND_PYTHON_PATH=${ASCEND_HOME}/python/site-packages
    export TBE_IMPL_PATH=${ASCEND_HOME}/opp/built-in/op_impl/ai_core/tbe
    export PYTHONPATH=${ASCEND_PYTHON_PATH}:${TBE_IMPL_PATH}:${PYTHONPATH}
}

function TryInstallWhl() {
    local base_path=${PWD}
    local arch=$(uname -m)
    if [[ ${arch} = "aarch64" ]]; then
        release_package_path=${base_path}/linux_aarch64/cloud_fusion/
    else
        release_package_path=${base_path}/centos_x86/cloud_fusion/
    fi

    # Install whl if available.
    ms_whl_path=`ls ${release_package_path}/mindspore-*.whl`
    mslite_whl_path=`ls ${release_package_path}/mindspore_lite-*.whl`
    base_path=$(pwd)

    if [[ -f "${ms_whl_path}" ]]; then
      pip uninstall mindspore -y || exit 1
      pip install ${ms_whl_path} --user || exit 1
      echo "install mindspore python whl success."
    fi

    if [[ -f "${mslite_whl_path}" ]]; then
      pip uninstall mindspore-lite -y || exit 1
      pip install ${mslite_whl_path} --user || exit 1
      echo "install mindspore_lite python whl success."
    fi
}

function RunAscendST() {
    echo "Run mindspore_lite python api st..."
    echo "-----------------------------------------------------------------------------------------"
    local base_path=${PWD}
    pytest ${base_path}/python/python_api/test_check_ascend.py -s
    RET=$?
    if [ ${RET} -ne 0 ]; then
      echo "Run test_check_ascend failed. Make sure mindspore/lite/python/api/_check_ascend.py:30"
      echo "compatible_cann_versions is updated correctly!"
      exit ${RET}
    fi
    echo "Run test_check_ascend success"
}

# Example:sh run_python_api_ascend.sh -r /home/temp_test -e x86_gpu
while getopts "r:e:" opt; do
    case ${opt} in
    r)
        release_path=${OPTARG}
        echo "release_path is ${release_path}"
        ;;
    ?)
        echo "unknown para"
        exit 1
        ;;
    esac
done


SetUpAscendEnv

TryInstallWhl

RunAscendST
