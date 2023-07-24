#!/bin/bash

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -e import_ms_and_mslite
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

echo "Run testcases of import mindspore and mindspore_lite..."
echo "-----------------------------------------------------------------------------------------"
cp ${models_path}/mobilenetv2.mindir ${basepath}

pytest -vra ${basepath}/python/import_ms_and_mslite/test_api_import_ms_and_mslite.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_api_import_ms_and_mslite failed."
  exit ${RET}
fi
echo "test_api_import_ms_and_mslite success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_api_import_mslite_and_ms.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_api_import_mslite_and_ms failed."
  exit ${RET}
fi
echo "test_api_import_mslite_and_ms success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_only_import_ms_and_mslite.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_only_import_ms_and_mslite failed."
  exit ${RET}
fi
echo "test_only_import_ms_and_mslite success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_only_import_mslite_and_ms.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_only_import_mslite_and_ms failed."
  exit ${RET}
fi
rm -rf ${basepath}/mobilenetv2.mindir
echo "test_only_import_mslite_and_ms success"

pytest -s ${basepath}/python/import_ms_and_mslite/test_predict_backend_lite_lenet.py --disable-warnings
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_predict_backend_lite_lenet failed."
  exit ${RET}
fi
echo "test_predict_backend_lite_lenet success"

pytest -s ${basepath}/python/import_ms_and_mslite/test_predict_backend_lite_resnet50.py --disable-warnings
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_predict_backend_lite_resnet50 failed."
  exit ${RET}
fi
echo "test_predict_backend_lite_resnet50 success"
