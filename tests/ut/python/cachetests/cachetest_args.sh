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

# source the globals and functions for use with cache testing
export SKIP_ADMIN_COUNTER=false
declare failed_tests
. cachetest_lib.sh
echo

################################################################################
# Cache testing: cache_admin argument testing                                  #
# Summary: Various tests that expect to get failure messages returned          #
################################################################################

# Double-command test
cmd="${CACHE_ADMIN} --start --stop"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# missing command test
cmd="${CACHE_ADMIN} --port 50082"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# bad arg test
cmd="${CACHE_ADMIN} -p abc --start"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# missing arg test
cmd="${CACHE_ADMIN} -p --start"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# invalid command
cmd="${CACHE_ADMIN} -p 50082 --start --not_exist_cmd"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# spill directory does not exist
cmd="${CACHE_ADMIN} --start --spilldir /path_that_does_not_exist"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# stop cache server first to test start
StopServer
# start cache server
StartServer
HandleRcExit $? 1 1
# start the cache server again, however, this time we expect an error
cmd="${CACHE_ADMIN} --start"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
StopServer
HandleRcExit $? 1 1

# start cache server twice with different ports
# this one starts with the default port 50052
StartServer
HandleRcExit $? 1 1
# this one starts with port 50053
cmd="${CACHE_ADMIN} --start -p 50053"
CacheAdminCmd "${cmd}" 0
HandleRcExit $? 1 1
# stop the cache server with default port
StopServer
HandleRcExit $? 1 1
# stop the cache server with port 50053
cmd="${CACHE_ADMIN} --stop -p 50053"
CacheAdminCmd "${cmd}" 0
HandleRcExit $? 1 1

# stop the cache server without bringing it up
cmd="${CACHE_ADMIN} --stop"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1

# start the cache server with illegal hostname
cmd="${CACHE_ADMIN} --start -h 0.0.0.0"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -h illegal"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -h"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -h --hostname"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -h --hostname 127.0.0.1"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1

# start the cache server with illegal port
cmd="${CACHE_ADMIN} --start -p 0"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -p -1"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -p 65536"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -p illegal"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1
cmd="${CACHE_ADMIN} --start -p"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 1

# find a port that is occupied using netstat
if [ -x "$(command -v netstat)" ]; then
  port=$(netstat -ntp | grep -v '::' | awk '{print $4}' | grep -E '^[[:digit:]]+' | awk -F: '{print $2}' | sort -n | tail -n 1)
  if [ ${port} -gt 1025 ]; then
    # start cache server with occupied port
    cmd="${CACHE_ADMIN} --start -p ${port}"
    CacheAdminCmd "${cmd}" 1
    HandleRcExit $? 0 1
  fi
fi

# generate session before starting the cache server
cmd="${CACHE_ADMIN} -g"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# illegal generate session command
StartServer
HandleRcExit $? 1 1
cmd="${CACHE_ADMIN} -g 1"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# illegal destroy session command
cmd="${CACHE_ADMIN} -d -2"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} -d illegal"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} -d"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
# destroy a non-existing session
cmd="${CACHE_ADMIN} -d 99999"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# stop cache server at this point
StopServer
HandleRcExit $? 1 1

# illegal number of workers
cmd="${CACHE_ADMIN} --start -w 0"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -w -1"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -w illegal"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
num_cpu=$(grep -c processor /proc/cpuinfo)
if [ $num_cpu -lt 100 ]; then
  cmd="${CACHE_ADMIN} --start -w 101"
else
  cmd="${CACHE_ADMIN} --start -w $(($num_cpu+1))"
fi
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -w 9999999"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -w"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# illegal spill path
cmd="${CACHE_ADMIN} --start -s"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

# spill path without writing perm
if [ "$EUID" -ne 0 ]; then
  cmd="${CACHE_ADMIN} --start -s /"
  CacheAdminCmd "${cmd}" 1
  HandleRcExit $? 0 0
fi

# illegal log level
cmd="${CACHE_ADMIN} --start -l 5"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -l -1"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0
cmd="${CACHE_ADMIN} --start -l"
CacheAdminCmd "${cmd}" 1
HandleRcExit $? 0 0

exit ${failed_tests}
