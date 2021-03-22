#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

CLANG_FORMAT=$(which clang-format) || (echo "Please install 'clang-format' tool first"; exit 1)

version=$("${CLANG_FORMAT}" --version | sed -n "s/.*\ \([0-9]*\)\.[0-9]*\.[0-9]*.*/\1/p")
if [[ "${version}" -lt "8" ]]; then
  echo "clang-format's version must be at least 8.0.0"
  exit 1
fi

CURRENT_PATH=$(pwd)
SCRIPTS_PATH=$(dirname "$0")

echo "CURRENT_PATH=$CURRENT_PATH"
echo "SCRIPTS_PATH=$SCRIPTS_PATH"

# print usage message
function usage()
{
  echo "Check whether the specified source files were well formatted"
  echo "Usage:"
  echo "bash $0 [-a] [-c] [-l] [-h]"
  echo "e.g. $0 -a"
  echo ""
  echo "Options:"
  echo "    -a Check code format of all files, default case"
  echo "    -c Check code format of the files changed compared to last commit"
  echo "    -l Check code format of the files changed in last commit"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  mode="all"    # default check all files

  # Process the options
  while getopts 'aclh' opt
  do
    case "${opt}" in
      a)
        mode="all"
        ;;
      c)
        mode="changed"
        ;;
      l)
        mode="lastcommit"
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

# init variable
# check options
checkopts "$@"

# switch to project root path, which contains clang-format config file '.clang-format'
cd "${SCRIPTS_PATH}/.." || exit 1

CHECK_LIST_FILE='__checked_files_list__'

if [ "X${mode}" == "Xall" ]; then
  find mindspore/{ccsrc,core,lite} -type f -name "*" | grep "\.h$\|\.cc$\|\.c$" > "${CHECK_LIST_FILE}" || true
elif [ "X${mode}" == "Xchanged" ]; then
  # --diff-filter=ACMRTUXB will ignore deleted files in commit
  git diff --diff-filter=ACMRTUXB --name-only | grep "mindspore/ccsrc\|mindspore/core\|mindspore/lite" | grep "\.h$\|\.cc$\|\.c$" > "${CHECK_LIST_FILE}" || true
else  # "X${mode}" == "Xlastcommit"
  git diff --diff-filter=ACMRTUXB --name-only HEAD~ HEAD | grep "mindspore/ccsrc\|mindspore/core\|mindspore/lite" | grep "\.h$\|\.cc$\|\.c$" > "${CHECK_LIST_FILE}" || true
fi

CHECK_RESULT_FILE=__code_format_check_result__
echo "0" > "$CHECK_RESULT_FILE"

# check format of files modified in the latest commit
while read line; do
  if [ ! -e ${line} ]; then
    continue
  fi
  BASE_NAME=$(basename "${line}")
  TEMP_FILE="__TEMP__${BASE_NAME}"
  cp "${line}" "${TEMP_FILE}"
  ${CLANG_FORMAT} -i "${TEMP_FILE}"
  diff "${TEMP_FILE}" "${line}"
  ret=$?
  rm "${TEMP_FILE}"
  if [[ "${ret}" -ne 0 ]]; then
    echo "File ${line} is not formatted, please format it."
    echo "1" > "${CHECK_RESULT_FILE}"
    break
  fi
done < "${CHECK_LIST_FILE}"

result=$(cat "${CHECK_RESULT_FILE}")
rm "${CHECK_RESULT_FILE}"
rm "${CHECK_LIST_FILE}"
cd "${CURRENT_PATH}" || exit 1
if [[ "X${result}" == "X0" ]]; then
  echo "Check PASS: specified files are well formatted!"
fi
exit "${result}"
