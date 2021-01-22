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
# Cache testing: cache cpp test driver                                         #
# Summary: A launcher for invoking cpp cache tests                             #
################################################################################

UT_TEST_DIR="${BUILD_PATH}/mindspore/tests/ut/cpp"
DateStamp=$(date +%Y%m%d_%H%M%S);
CPP_TEST_LOG_OUTPUT="/tmp/ut_tests_cache_${DateStamp}.log"

# start cache server with a spilling path to be used for all tests
cmd="${CACHE_ADMIN} --start -s /tmp"
CacheAdminCmd "${cmd}" 0
sleep 1
HandleRcExit $? 1 1

# Set the environment variable to enable these pytests
export RUN_CACHE_TEST=TRUE
GTEST_FILTER_OLD=$GTEST_FILTER
export GTEST_FILTER="MindDataTestCacheOp.*"
export GTEST_ALSO_RUN_DISABLED_TESTS=1

# All of the cpp tests run under the same session
GetSession
HandleRcExit $? 1 1
export SESSION_ID=$session_id

test_count=$(($test_count+1))
cd ${UT_TEST_DIR} 
cmd="${UT_TEST_DIR}/ut_tests"
echo "Test ${test_count}: ${cmd}"
MsgEnter "Run test ${test_count}"
${cmd} > ${CPP_TEST_LOG_OUTPUT} 2>&1
rc=$?
if [ ${rc} -ne 0 ]; then
   MsgFail "FAILED"
   MsgError "Invoking cpp tests failed!" "${rc}" "See log: ${CPP_TEST_LOG_OUTPUT}"
else
   MsgOk "OK"
fi
echo
HandleRcExit $rc 1 0

cd ${CURRPATH}

StopServer
HandleRcExit $? 1 0

# restore old env var
export GTEST_FILTER=$GTEST_FILTER_OLD
unset RUN_CACHE_TEST
unset GTEST_ALSO_RUN_DISABLED_TESTS

exit ${failed_tests}
