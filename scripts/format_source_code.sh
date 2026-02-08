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

echo "CURRENT_PATH=${CURRENT_PATH}"
echo "SCRIPTS_PATH=${SCRIPTS_PATH}"

# print usage message
function usage()
{
  echo "Format the specified source files to conform the code style."
  echo "Usage:"
  echo "bash $0 [-a] [-c] [-l] [-h]"
  echo "e.g. $0 -c"
  echo ""
  echo "Options:"
  echo "    -a format of all files"
  echo "    -c format of the files changed compared to last commit, default case"
  echo "    -l format of the files changed in last commit"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  mode="changed"    # default format changed files

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

FMT_FILE_LIST='__format_files_list__'

if [[ "X${mode}" == "Xall" ]]; then
  find mindspore/{ccsrc,core,lite} -type f -name "*" | grep -E "\.(h|cc|c)$" > "${FMT_FILE_LIST}" || true
elif [[ "X${mode}" == "Xchanged" ]]; then
  git diff --name-only | grep "mindspore/ops\|mindspore/ccsrc\|mindspore/core\|mindspore/lite\|include" | grep -E "\.(h|cc|c)$" > "${FMT_FILE_LIST}" || true
else  # "X${mode}" == "Xlastcommit"
  git diff --name-only HEAD~ HEAD | grep "mindspore/ops\|mindspore/ccsrc\|mindspore/core\|mindspore/lite\|include" | grep -E "\.(h|cc|c)$" > "${FMT_FILE_LIST}" || true
fi

while read line; do
  if [ -f "${line}" ]; then
    "${CLANG_FORMAT}" -i "${line}"
  fi
done < "${FMT_FILE_LIST}"

rm "${FMT_FILE_LIST}"
cd "${CURRENT_PATH}" || exit 1

echo "Specified cpp source files have been format successfully."
