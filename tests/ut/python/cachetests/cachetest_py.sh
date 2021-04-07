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
export SKIP_ADMIN_COUNTER=true
declare session_id failed_tests
. cachetest_lib.sh
echo

################################################################################
# Cache testing: cache python test driver                                      #
# Summary: Various tests for running the python testcases for caching          #
################################################################################

StartServer
HandleRcExit $? 1 1

# Set the environment variable to enable these pytests
export RUN_CACHE_TEST=TRUE

# Each of these tests will create session, use it, then destroy it after the test
for i in $(seq 1 5)
do
   test_name="test_cache_map_basic${i}"
   GetSession
   HandleRcExit $? 1 1
   export SESSION_ID=$session_id

   PytestCmd "test_cache_map.py" "${test_name}"
   HandleRcExit $? 0 0

   DestroySession $session_id
   HandleRcExit $? 1 1
done

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

# use pytest pattern match to run all the tests that match the name test_cache_map_failure.
# All of these tests will interact with the same cache session and may result in multiple
# caches under the common session handle (although these are failure tests so probably not)
PytestCmd "test_cache_map.py" "test_cache_map_failure" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_split" 1
HandleRcExit $? 0 0

# DatasetCache parameter check
PytestCmd "test_cache_map.py" "test_cache_map_parameter_check"
HandleRcExit $? 0 0

# Executing the same pipeline for twice under the same session
# Executing the same pipeline for twice (from python)
PytestCmd "test_cache_map.py" "test_cache_map_running_twice1"
HandleRcExit $? 0 0
# Executing the same pipeline for twice (from shell)
PytestCmd "test_cache_map.py" "test_cache_map_running_twice2"
HandleRcExit $? 0 0
PytestCmd "test_cache_map.py" "test_cache_map_running_twice2"
HandleRcExit $? 0 0

# Executing the same pipeline for twice under the different session
# Executing the same pipeline for twice (from shell)
PytestCmd "test_cache_map.py" "test_cache_map_running_twice2"
HandleRcExit $? 0 0
DestroySession $session_id
HandleRcExit $? 1 1
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id
PytestCmd "test_cache_map.py" "test_cache_map_running_twice2"
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_no_image"
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_parallel_workers"
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_num_connections" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_prefetch_size" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_to_device"
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_epoch_ctrl" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_coco" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_mnist" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_celeba" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_manifest" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_cifar" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_voc" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_python_sampler" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_map.py" "test_cache_map_nested_repeat"
HandleRcExit $? 0 0

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_map.py" "test_cache_map_interrupt_and_rerun"
HandleRcExit $? 0 0

DestroySession $session_id
HandleRcExit $? 1 1

# Run two parallel pipelines (sharing cache)
for i in $(seq 1 2)
do
   test_name="test_cache_map_parallel_pipeline${i}"
   GetSession
   HandleRcExit $? 1 1
   export SESSION_ID=$session_id

   PytestCmd "test_cache_map.py" "${test_name} --shard 0" &
   pids+=("$!")
   PytestCmd "test_cache_map.py" "${test_name} --shard 1" &
   pids+=("$!")

   for pid in "${pids[@]}"; do
      wait ${pid}
      HandleRcExit $? 0 0
   done

   # Running those PytestCmd in the background will not get our test_count updated. So we need to manually update it here.
   test_count=$(($test_count+1))
   DestroySession $session_id
   HandleRcExit $? 1 1
done

StopServer
HandleRcExit $? 1 1
sleep 1

# test cache server with --workers 1
cmd="${CACHE_ADMIN} --start --workers 1"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 0 0

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_map.py" "test_cache_map_server_workers_1"
HandleRcExit $? 0 0
StopServer
HandleRcExit $? 0 1

# test cache server with --workers 100
cmd="${CACHE_ADMIN} --start --workers 100"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 0 0

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_map.py" "test_cache_map_server_workers_100"
HandleRcExit $? 0 0
StopServer
HandleRcExit $? 0 1

# The next set of testing is for the non-mappable cases.
StartServer
HandleRcExit $? 1 1

# This runs all of the basic tests.  These will all share the same and we do not destroy
# the session in between each.
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_basic" 1
HandleRcExit $? 0 0

DestroySession $session_id
HandleRcExit $? 1 1

# run the small shared cache tests
for i in $(seq 1 4)
do
   test_name="test_cache_nomap_allowed_share${i}"
   GetSession
   HandleRcExit $? 1 1
   export SESSION_ID=$session_id

   PytestCmd "test_cache_nomap.py" "${test_name}"
   HandleRcExit $? 0 0

   DestroySession $session_id
   HandleRcExit $? 1 1
done

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_disallowed_share" 1
HandleRcExit $? 0 0

DestroySession $session_id
HandleRcExit $? 1 1

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

# Executing the same pipeline for twice under the same session
# Executing the same pipeline for twice (from python)
PytestCmd "test_cache_nomap.py" "test_cache_nomap_running_twice1"
HandleRcExit $? 0 0
# Executing the same pipeline for twice (from shell)
PytestCmd "test_cache_nomap.py" "test_cache_nomap_running_twice2"
HandleRcExit $? 0 0
PytestCmd "test_cache_nomap.py" "test_cache_nomap_running_twice2"
HandleRcExit $? 0 0

# Executing the same pipeline for twice under the different session
# Executing the same pipeline for twice (from shell)
PytestCmd "test_cache_nomap.py" "test_cache_nomap_running_twice2"
HandleRcExit $? 0 0
DestroySession $session_id
HandleRcExit $? 1 1
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id
PytestCmd "test_cache_nomap.py" "test_cache_nomap_running_twice2"
HandleRcExit $? 0 0

# Run two parallel pipelines (sharing cache)
for i in $(seq 1 2)
do
   test_name="test_cache_nomap_parallel_pipeline${i}"
   GetSession
   HandleRcExit $? 1 1
   export SESSION_ID=$session_id

   PytestCmd "test_cache_nomap.py" "${test_name} --shard 0" &
   pids+=("$!")
   PytestCmd "test_cache_nomap.py" "${test_name} --shard 1" &
   pids+=("$!")
   PytestCmd "test_cache_nomap.py" "${test_name} --shard 2" &
   pids+=("$!")

   for pid in "${pids[@]}"; do
      wait ${pid}
      HandleRcExit $? 0 0
   done

   # Running those PytestCmd in the background will not get our test_count updated. So we need to manually update it here.
   test_count=$(($test_count+1))
   DestroySession $session_id
   HandleRcExit $? 1 1
done

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_parallel_workers"
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_num_connections" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_prefetch_size" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_to_device"
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_epoch_ctrl" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_clue" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_csv" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_textfile" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_nested_repeat"
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_get_repeat_count"
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_long_file_list"
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_failure" 1
HandleRcExit $? 0 0

PytestCmd "test_cache_nomap.py" "test_cache_nomap_pyfunc" 1
HandleRcExit $? 0 0

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_all_rows_cached"
HandleRcExit $? 0 0

DestroySession $session_id
HandleRcExit $? 1 1

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_interrupt_and_rerun"
HandleRcExit $? 0 0

DestroySession $session_id
HandleRcExit $? 1 1

for i in $(seq 1 3)
do
   test_name="test_cache_nomap_multiple_cache${i}"
   GetSession
   HandleRcExit $? 1 1
   export SESSION_ID=$session_id

   PytestCmd "test_cache_nomap.py" "${test_name}"
   HandleRcExit $? 0 0

   DestroySession $session_id
   HandleRcExit $? 1 1
done

# Create session, run train and eval pipeline concurrently with different cache
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id
PytestCmd "test_cache_nomap.py" "test_cache_nomap_multiple_cache_train" &
pids+=("$!")
PytestCmd "test_cache_nomap.py" "test_cache_nomap_multiple_cache_eval" &
pids+=("$!")

for pid in "${pids[@]}"; do
   wait ${pid}
   HandleRcExit $? 0 0
done

# Running those PytestCmd in the background will not get our test_count updated. So we need to manually update it here.
test_count=$(($test_count+1))
DestroySession $session_id
HandleRcExit $? 1 1

# Create session, use it to run a pipeline, and destroy the session while pipeline is running
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_session_destroy" &
pid=$!

sleep 10
DestroySession $session_id
HandleRcExit $? 1 1
wait ${pid}
# Running those PytestCmd in the background will not get our test_count updated. So we need to manually update it here.
test_count=$(($test_count+1))

# Stop cache server while pipeline is running
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

PytestCmd "test_cache_nomap.py" "test_cache_nomap_server_stop" &
pid=$!

sleep 10
StopServer
HandleRcExit $? 1 1
sleep 1
wait ${pid}
# Running those PytestCmd in the background will not get our test_count updated. So we need to manually update it here.
test_count=$(($test_count+1))

# test cache server with --workers 1
cmd="${CACHE_ADMIN} --start --workers 1"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 0 0
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id
PytestCmd "test_cache_nomap.py" "test_cache_nomap_server_workers_1"
HandleRcExit $? 0 0
StopServer
HandleRcExit $? 0 1

# test cache server with --workers 100
cmd="${CACHE_ADMIN} --start --workers 100"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 0 0
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id
PytestCmd "test_cache_nomap.py" "test_cache_nomap_server_workers_100"
HandleRcExit $? 0 0
StopServer
HandleRcExit $? 0 1

# start cache server with a spilling path
cmd="${CACHE_ADMIN} --start -s /tmp"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 0 0

GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

# Set size parameter of mappable DatasetCache to a extra small value
PytestCmd "test_cache_map.py" "test_cache_map_extra_small_size" 1
HandleRcExit $? 0 0
# Set size parameter of non-mappable DatasetCache to a extra small value
PytestCmd "test_cache_nomap.py" "test_cache_nomap_extra_small_size" 1
HandleRcExit $? 0 0

StopServer
HandleRcExit $? 0 1

unset RUN_CACHE_TEST
unset SESSION_ID

exit ${failed_tests}
