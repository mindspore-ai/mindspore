#!/bin/bash

# Example:sh run_benchmark_nets.sh -r /home/temp_test -e plugin_custom_ops
while getopts "r:m:e:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${release_path}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${models_path}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${level}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

ms_whl_path=`ls ${release_path}/mindspore-*.whl`
mslite_whl_path=`ls ${release_path}/mindspore_lite-*.whl`
basepath=$(pwd)

if [[ -f "${ms_whl_path}" ]]; then
  pip uninstall mindspore -y || exit 1
  pip install ${ms_whl_path} --user || exit 1
  echo "install mindspore python whl success."
else
  echo "not find mindspore python whl.."
  exit 1
fi

if [[ -f "${mslite_whl_path}" ]]; then
  pip uninstall mindspore-lite -y || exit 1
  pip install ${mslite_whl_path} --user || exit 1
  echo "install mindspore_lite python whl success."
else
  echo "not find mindspore_lite python whl.."
  exit 1
fi

echo "Run testcases of mindspore_lite plugin custom ops..."
echo "-----------------------------------------------------------------------------------------"

python ${basepath}/python/plugin_custom_ops/test_kv_cache_mgr.py Ascend
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_kv_cache_mgr failed."
  exit ${RET}
fi
echo "test_plugin_custom_ops success"
