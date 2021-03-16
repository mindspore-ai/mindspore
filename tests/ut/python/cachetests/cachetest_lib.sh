#!/bin/bash
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

# This file is a collection of functions and globals that are common to the
# test scenarios for cache op testing.

# Set any path variables here
CURRPATH=$(cd "$(dirname $0)"; pwd)
TESTPATH=$(cd "${CURRPATH}/../dataset"; pwd)
PROJECT_PATH=$(cd "${CURRPATH}/../../../../"; pwd)

if [ "x${BUILD_PATH}" == "x" ]; then
   BUILD_PATH=${PROJECT_PATH}/build
fi
echo "Using test path: ${TESTPATH}"
echo "Using build path: ${BUILD_PATH}"

# Point to the cache_admin from the build path.  The user may also have installed the wheel file but we don't know that.
CACHE_ADMIN="${BUILD_PATH}/package/mindspore/bin/cache_admin"
PYTHON_PYTEST="python -m pytest ${TESTPATH}/"

# These are globals that all testcases use and may get updated during testcase running
failed_tests=0
test_count=0
session_id=0

# sanity check on the cache_admin

if [ ! -f ${CACHE_ADMIN} ]; then
   echo "Could not find cache_admin binary. ${CACHE_ADMIN}"
   exit 1
fi

#################################################################################
#  Function: MsgEnter                                                           #
#  Description: Display the leading text before entering a block of logic.      #
#################################################################################
MsgEnter()
{
   printf "%-60s : " "${1}"        
}

#################################################################################
#  Function: MsgOK                                                              #
#  Description: Display input msg with a green format for success               #
#################################################################################
MsgOk()
{
   echo -e '\E[32m'"\033[1m$1\033[0m"
}

#################################################################################
#  Function: MsgFail                                                            #
#  Description: Display input msg with a red format for a failure              #
#################################################################################
MsgFail()
{
   echo -e '\E[31m'"\033[1m$1\033[0m"
}

#################################################################################
#  Function: MsgError                                                           #
#  Description: If something is not successful, display some info about it      #
#                                                                               #
#  Arguments are optional with defaults.  You should pass empty string for any  #
#  args not being used so that it chooses the defaults.                         #
#                                                                               #
#  Optional arguments:  arg 1: An error message.                                #
#                       arg 2: The return code.                                 #
#                       arg 3: The error details                                #
#                                                                               #
#################################################################################
MsgError()
{
   msg=${1:-none}
   err_rc=${2:-none}
   err_detail=${3:-none}

   if [ "${msg}" != "none" ] ; then
      echo "${msg}"
   fi

   if [ "${err_rc}" != "none" ] ; then
      echo "Return code: ${err_rc}"
   fi

   if [ "${err_detail}" != "none" ] ; then
      echo "Error detail:"
      echo "{$err_detail}"
   fi
   echo
}

#################################################################################
#  Function: ServerCleanup                                                      #
#  Description: This is a non-code method to clean up a running cache server.   #
#               The intended use is for cases when some command has failed, we  #
#               want to check for any stale process or resources and forcefully #
#               remove those resources.                                         #
#################################################################################
ServerCleanup()
{
   echo "ServerCleanup is running"
   server_pid=$(ps -elf | grep ${USER} | grep cache_server | grep -v grep | awk '{print $4}')
   if [ "x${server_pid}" != "x" ]; then
      echo "Found a running cache server pid ${server_pid}.  Killing this process"
      kill -9 ${server_pid}
      # small sleep to allow some cleanup time
      sleep 2
   fi

   for i in `ipcs -m | grep ${USER} | awk '{print $2}'`
   do
      ipcrm -m ${i}
   done

   echo "ServerCleanup complete."
}

#################################################################################
#  Function: CacheAdminCmd                                                      #
#  Description: Wrapper function for executing cache_admin commands             #
#               Caller must use HandleRcExit to process the return code.        #
#                                                                               #
#  Arguments:   arg 1: The command to run                                       #
#               arg 2: value of 0 means that we check rc for success. a value   #
#                      of 1 means that we expect a failure from the command.    #
#################################################################################
CacheAdminCmd()
{
   if [ $# -ne 2 ]; then
      echo "Test script invalid.  Bad CacheAdminCmd function args."
      exit 1
   fi
   cmd=$1
   expect_fail=$2
   if [ "${SKIP_ADMIN_COUNTER}" != "true" ]; then
      test_count=$(($test_count+1))
      echo "Test ${test_count}: ${cmd}"
      MsgEnter "Run test ${test_count}"
   fi
   result=$(${cmd} 2>&1)
   rc=$?
   if [ "${expect_fail}" -eq 0 ] && [ "${rc}" -ne 0 ]; then
      MsgFail "FAILED"
      MsgError "cache_admin command failure!" "${rc}" "${result}"
      return 1
   elif [ "${expect_fail}" -eq 1 ] && [ "${rc}" -eq 0 ]; then
      MsgFail "FAILED"
      MsgError "Expected failure but got success!" "${rc}" "${result}"
      return 1
   else
      if [ "${SKIP_ADMIN_COUNTER}" != "true" ]; then
         MsgOk "OK"
      fi
   fi
   echo
   return 0
}

#################################################################################
#  Function: PytestCmd                                                          #
#  Description: Wrapper function for executing pytest                           #
#               Caller must use HandleRcExit to process the return code.        #
#                                                                               #
#  Arguments:   arg 1: The python script name                                   #
#               arg 2: The python function name                                 #
#################################################################################
PytestCmd()
{
   test_count=$(($test_count+1))
   py_script=$1
   py_func=$2
   pattern=${3:-0}
   # python scripts require special relative paths
   cd ..
   if [ ${pattern} -eq 0 ]; then
      cmd="${PYTHON_PYTEST}${py_script}::${py_func}"
   elif [ ${pattern} -eq 1 ]; then
      cmd="${PYTHON_PYTEST}${py_script} -k ${py_func}"
   else
      echo "Invalid Pytest command test script error"
      exit 1
   fi
   echo "Test ${test_count}: ${cmd}"
   MsgEnter "Run test ${test_count}"
   result=$(${cmd} 2>&1)
   rc=$?
   if [ ${rc} -ne 0 ]; then
      MsgFail "FAILED"
      MsgError "pytest call had failure!" "${rc}" "${result}"
      cd ${CURRPATH}
      return 1
   else
      MsgOk "OK"
   fi
   echo
   cd ${CURRPATH}
   return 0
}

#################################################################################
#  Function: StartServer                                                        #
#  Description: Helper function to call cache_admin to start a default server   #
#               Caller must use HandleRcExit to process the return code.        #
#################################################################################
StartServer()
{
   cmd="${CACHE_ADMIN} --start"
   CacheAdminCmd "${cmd}" 0
   sleep 1
   return $?
}

#################################################################################
#  Function: StopServer                                                         #
#  Description: Helper function to call cache_admin to stop cache server        #
#               Caller must use HandleRcExit to process the return code.        #
#################################################################################
StopServer()
{
   cmd="${CACHE_ADMIN} --stop"
   CacheAdminCmd "${cmd}" 0
   return $?
}

#################################################################################
#  Function: GetSession                                                         #
#  Description: Helper function to call cache_admin to generate a session       #
#               Caller must use HandleRcExit to process the return code.        #
#################################################################################
GetSession()
{
   # Cannot use CacheAdminCmd for this one because we have special action to set
   # the global variable for session id.
   cmd="${CACHE_ADMIN} --generate_session"
   if [ "${SKIP_ADMIN_COUNTER}" != "true" ]; then
      test_count=$(($test_count+1))
      echo "Test ${test_count}: ${cmd}"
      MsgEnter "Run test ${test_count}"
   fi
   result=$(${cmd} 2>&1)
   rc=$?
   if [ ${rc} -ne 0 ]; then
      MsgFail "FAILED"
      MsgError "cache_admin command failure!" "${rc}" "${result}"
      return 1
   else
      session_id=$(echo $result | awk '{print $NF}')
      if [ "${SKIP_ADMIN_COUNTER}" != "true" ]; then
         MsgOk "OK"
         echo "Generated session id:  ${session_id}"
         echo
      fi
   fi
   return 0
}

#################################################################################
#  Function: DestroySession                                                     #
#  Description: Helper function to call cache_admin to destroy a session        #
#               Caller must use HandleRcExit to process the return code.        #
#################################################################################
DestroySession()
{
   cmd="${CACHE_ADMIN} --destroy_session ${session_id}"
   CacheAdminCmd "${cmd}" 0
   return $?
}

#################################################################################
#  Function: HandlerRcExit                                                      #
#  Description: handles a return code if you used one of the above helper funcs #
#               It updates the global test counters and chooses to quit or not  #
#               depending on the setting of exit_on_fail argument               #
#                                                                               #
#  Arguments:   arg 1: The rc to handle                                         #
#               arg 2: Set to 1 to cause error exit.  0 for no exit             #
#               arg 3: Set to 1 to invoke server cleanup on error case          #
#################################################################################
HandleRcExit()
{
   if [ $# -ne 3 ]; then
      echo "Test script invalid.  Bad CacheAdminCmd function args."
      exit 1
   fi
    
   err_rc=$1
   exit_on_fail=$2
   clean_on_fail=$3

   if [ ${err_rc} -ne 0 ]; then
      failed_tests=$(($failed_tests+1))
       
      if [ ${clean_on_fail} -eq 1 ]; then
          ServerCleanup
      fi

      if [ ${exit_on_fail} -eq 1 ]; then
         exit $failed_tests
      else
         return 1
      fi
   fi

   return 0
}

#################################################################################
#  Function: ExitHandler                                                        #
#  Description: Invokes final display message of the script before quitting     #
#################################################################################
ExitHandler()
{
   success_count=$(($test_count-$failed_tests))
   echo "------------------------------------"
   echo "${test_count} tests run in total."
   echo "${success_count} tests ran successfully."
   echo "${failed_tests} failed tests."
   exit ${failed_tests}
}

trap ExitHandler EXIT SIGINT
