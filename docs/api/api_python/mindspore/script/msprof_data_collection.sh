#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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

# This script is used to run perf/iotop/ltrace by profiling.

command_type=$1
command_param=$2
script_dir="/usr/bin"
script_name="$(basename "$0")"
reg_int='^[1-9][0-9]{,6}$|^0$'

function get_version(){
    if [ "${command_param}" = "perf" ] || [ "${command_param}" = "ltrace" ] || [ "${command_param}" = "iotop" ]; then
        "${command_param}" --version
    else
        printf "The value of the second parameter is incorrect, please enter the correct parameter, "
        printf "such as: perf, ltrace, iotop\n"
        exit 1
    fi
}

function kill_prof_cmd(){
    if [[ ${command_param} =~ ${reg_int} ]]; then
        ppid=`ps -o ppid= -p ${command_param}`
        ppid_user=$(ps -o uid -e -o pid | awk -va="${ppid}" '$2==a {print $1}')
        shell_user=`id -u ${SUDO_USER}`
        if [ "${ppid_user}" != "${shell_user}" ]; then
            echo "UID of ${ppid} is:${ppid_user}, UID running this script is:${shell_user}"
            exit 1
        fi
        pidLine=`pstree -p ${command_param}`
        pidLine=`echo $pidLine | awk 'BEGIN{ FS="(" ; RS=")" } NF>1 { print $NF }'`
        for pid in $pidLine
            do
                if [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]]; then
                    if kill -0 "${pid}" 2>/dev/null; then
                        sudo kill -9 "${pid}"
                    fi
                fi
            done     
        exit 1
    else
        echo "Input pid:${command_param} error"
        exit 1
    fi
}

# The user currently running this script and the user to which the PID process belongs must be consistent.
function check_pid(){
    if [[ ! ${command_param} =~ ${reg_int} ]]; then
        echo "Input pid:${command_param} error"
        exit 1
    fi
    params=$(cat /proc/sys/kernel/pid_max)
    if [[ ! "$params" =~ ${reg_int} ]]; then
        echo "Get max_pid error"
        exit 1
    fi
    if [ "${command_param}" -gt "${params}" ]; then
        echo "Input pid:${command_param} gt pid_max:${params}"
        exit 1
    fi
    pid_user=$(ps -o uid -e -o pid | awk -va="${command_param}" '$2==a {print $1}')
    shell_user=`id -u ${SUDO_USER}`
    if [ "${pid_user}" != "${shell_user}" ]; then
        echo "UID of ${command_param} is:${pid_user}, UID running this script is:${shell_user}"
        exit 1
    fi
}

function run_prof_trace_cmd(){
    check_pid
    local max_retries=20
    local status=-1
    for ((retry=0; retry < max_retries && status != 0; retry++)); do
        perf trace -T --syscalls -p "${command_param}"
        status=$?
    done
}

function run_ltrace_cmd(){
    check_pid
}

function run_iotop_cmd(){
    check_pid
    iotop -b -d 0.02 -P -t -p "${command_param}" &
    while ps -p "${command_param}" > /dev/null 2>&1; do
        sleep 1
    done
    pkill -9 iotop
}

function check_username(){
    echo "${command_param}" | grep -q -E '^[ 0-9a-zA-Z./:]*$'
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "Parameter:${command_param} is invalied!"
        exit 1
    fi
    if ! id -u "${command_param}" >/dev/null 2>&1 ; then
        echo "User:${command_param} does not exist"
        exit 1
    fi
}

function get_cmd(){
    params=$(cat /proc/sys/kernel/pid_max)
    if [[ ! "$params" =~ ${reg_int} ]]; then
        echo "Get max_pid error"
        exit 1
    fi
    digits=1
    while ((${params}>10)); do
        let "digits++"
        ((params /= 10))
    done
    compile='[1-9]'
    arr[0]='[0-9]'
    for((i=1;i<digits;i++)); do
        compile="${compile}[0-9]"
        arr[i]=${compile}
    done
    cmd="${script_dir}/${script_name} get-version perf,${script_dir}/${script_name} get-version ltrace,${script_dir}/${script_name} get-version iotop"
    for i in "${arr[@]}"; do
        cmd="${cmd},${script_dir}/${script_name} perf $i,${script_dir}/${script_name} ltrace $i,${script_dir}/${script_name} iotop $i,${script_dir}/${script_name} kill $i"
    done
    cmd="$command_param ALL=(ALL:ALL) NOPASSWD:${cmd}"
    cmd=$(echo -e "${cmd}\nDefaults env_reset")
    echo "${cmd}"
}

function set_sudoers(){
    if [ -d "/etc/sudoers.d" ]; then
        if [ -f "/etc/sudoers.d/${command_param}_profiling" ]; then
            echo "The file /etc/sudoers.d/${command_param}_profiling already exist"
        fi
        echo "${cmd}" > /etc/sudoers.d/"${command_param}"_profiling
        result=$?
        if [ "$result" -ne 0 ]; then
            echo "Set cmd to /etc/sudoers.d/${command_param}_profiling failed!"
            exit 1
        else
            echo "The user permission have been configured successfully. You can find the configuration file /etc/sudoers.d/${command_param}_profiling"
            exit
        fi
    fi
    has_add=$(cat /etc/sudoers|grep "${script_name}"|grep "${command_param}")
    if [ "${has_add}" ]; then
        echo "The configure already exist, please confirm its content is correct"
        exit
    fi
    chmod u+w /etc/sudoers
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "Permission configure failed"
        exit 1
    fi
    echo "${cmd}" >> /etc/sudoers
    chmod u-w /etc/sudoers
    echo "The user permission have been configured successfully. You can find the configuration file in the /etc/sudoers."
}

function handle_sudoers(){
    check_username
    get_cmd
    set_sudoers
}

function main(){
    if [ $# -ne 2 ]; then
        echo "The number of parameters is incorrect, please enter two parameters"
        exit 1
    fi
    if [ "${command_type}" = "set-sudoers" ]; then
        echo "Run set-sudoers cmd"
        handle_sudoers
    elif [ "${command_type}" = "get-version" ]; then
        #echo "Run get-version cmd"
        get_version
    elif [ "${command_type}" = "kill" ]; then
        #echo "kill cmd"
        kill_prof_cmd
    elif [ "${command_type}" = "perf" ]; then
        #echo "run perf trace cmd"
        run_prof_trace_cmd
    elif [ "${command_type}" = "ltrace" ] ; then
        #echo "run ltrace cmd"
        run_ltrace_cmd
    elif [ "${command_type}" = "iotop" ]; then
        #echo "run iotop cmd"
        run_iotop_cmd
    else
        printf "The value of the first parameter is incorrect, please enter the correct parameter, "
        printf "such as: set-sudoers, get-version, kill, perf, ltrace, iotop\n"
        exit 1
    fi
}

main "$@"
