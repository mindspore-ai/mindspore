#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
bootup_cache_server()
{
  echo "Booting up cache server..."
  result=$(cache_admin --start 2>&1)
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ] && [[ ! ${result} =~ "Cache server is already up and running" ]]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}

generate_cache_session()
{
  result=$(cache_admin -g | awk 'END {print $NF}')
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}

shutdown_cache_server()
{
  echo "Shutting down cache server..."
  result=$(cache_admin --stop 2>&1)
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ] && [[ ! ${result} =~ "Server on port 50052 is not reachable or has been shutdown already" ]]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}
