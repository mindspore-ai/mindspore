#!/bin/bash

# Usage:
#    Command: sh run_st.sh -p ascend -c ascend910
#    Options:
#        -p: platform type(ascend, gpu, all)
#        -c: driver version(cuda-10.1, cuda-11.1 ascend910, ascend310)
#        -l: testcase level(level0, level1, level2, level3)
#        -r: testcase root path

set -e

# Check parameter
# bool_empty: whether the parameter is allowed to be empty. Default: true
function CHECK_PARAMETER() {
    local current_param=${1}
    local param_name=${2}
    local param_list=${3}
    local bool_empty=${4:-"true"}
    local flag_check=1
    local param=""

    # If in the parameter list
    for param in ${param_list//,/ }; do
        if [ "${current_param}" = "${param}" ]; then
            flag_check=0
            break
        fi
    done

    # If empty
    if [ "${bool_empty}" = "true" ] && [ -z "${current_param}" ]; then
        flag_check=0
    fi

    # Check result
    if [ "${flag_check}" = "1" ];then
        echo "Input parameter of ${param_name} is invalid. (Value: ${param_list})"
        exit 1
    fi
}

########
# Init #
########
# Global parameter
CURR_DIR=$(dirname "${BASH_SOURCE-$0}")
CURR_DIR=$(cd -P "${CURR_DIR}"; pwd -P)
CONFIG_PATH="${CURR_DIR}/config"
TESTCASE_LEVEL='level0'
TESTCASE_ROOT="${CURR_DIR}/../"

# Get input parameter
while getopts ':p:c:l:r:' opt; do
    case "${opt}" in
        p)
          PLATFORM_TYPE=${OPTARG}
          ;;
        c)
          DRIVER_VERSION=${OPTARG}
          ;;
        l)
          TESTCASE_LEVEL=${OPTARG}
          ;;
        r)
          TESTCASE_ROOT=${OPTARG}
          ;;
        *)
          echo "[ERROR] Unknown option ${opt}."
          exit 1
          ;;
    esac
done

# Check parameter
CHECK_PARAMETER "${PLATFORM_TYPE}" "platform_type" "ascend, gpu, all" "false"

##########
# Config #
##########
ENV_TYPE="ASCEND_ARM_EULEROS"

# Set config(case_env_config.yaml)
cat ${CONFIG_PATH}/case_env_config_template.yaml|grep -A 8 "all_case_type" > ${CONFIG_PATH}/case_env_config.yaml
echo -e "\ncase_type:" >> ${CONFIG_PATH}/case_env_config.yaml
if [ "${PLATFORM_TYPE}" = "ascend" ] || [ "${PLATFORM_TYPE}" = "all" ]; then
    ENV_TYPE="ASCEND_ARM_EULEROS"
    echo -e "  platform_arm_ascend_training: ASCEND_ARM_EULEROS" >> ${CONFIG_PATH}/case_env_config.yaml
elif [ "${PLATFORM_TYPE}" = "ascend910b" ] || [ "${PLATFORM_TYPE}" = "all" ]; then
    ENV_TYPE="ASCEND_ARM_EULEROS_910B"
    echo -e "  platform_arm_ascend910b_training: ASCEND_ARM_EULEROS_910B" >> ${CONFIG_PATH}/case_env_config.yaml
elif [ "${PLATFORM_TYPE}" = "gpu" ] || [ "${PLATFORM_TYPE}" = "all" ]; then
    if [ "${DRIVER_VERSION}" = "cuda-10.1" ]; then
        ENV_TYPE="GPU_X86_UBUNTU_CUDA10"
        echo -e "  platform_x86_gpu_training: GPU_X86_UBUNTU_CUDA10" >> ${CONFIG_PATH}/case_env_config.yaml
    else
        ENV_TYPE="GPU_X86_UBUNTU_CUDA11"
        echo -e "  platform_x86_gpu_training: GPU_X86_UBUNTU_CUDA11" >> ${CONFIG_PATH}/case_env_config.yaml
    fi
fi


#################
# Run testcases #
#################
# Get parameter
parameter_list="--case_root ${TESTCASE_ROOT} --filter_keyword ${TESTCASE_LEVEL}"

# Run testcase
cd ${CURR_DIR}
python3 -u run_st.py ${parameter_list} --env_config_file ${CONFIG_PATH}/env_config.yaml \
    --case_env_config_path ${CONFIG_PATH}/case_env_config.yaml --env_type $ENV_TYPE
