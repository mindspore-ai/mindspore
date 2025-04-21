#!/bin/bash
vendor_name=customize
script_dir=$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)
sourcedir=${script_dir}/packages
vendordir=vendors/$vendor_name
QUIET="y"


log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[runtime] [$cur_date] "$1
}

handle()
{
    targetdir=$2

    if [ ! -d ${targetdir}/$vendor_name/$1 ];then
        log "[INFO] create ${targetdir}/$vendor_name/$1."
        mkdir -p ${targetdir}/$vendor_name/$1
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendor_name/$1 failed"
            return 1
        fi
    else
        has_same_file=-1
        for file_a in ${sourcedir}/$vendordir/$1/*; do
            file_b=${file_a##*/};
        if [ ! "$(ls -A ${targetdir}/$vendor_name/$1)" ]; then
                log "[INFO] ${targetdir}/$vendor_name/$1 is empty !!"
                return 1
        fi
            # shellcheck disable=SC2046
            grep -q $file_b <<<'ls ${targetdir}/$vendor_name/$1';
            if [[ $? -eq 0 ]]; then
                echo -n "${file_b} "
                has_same_file=0
            fi
        done
        if [ 0 -eq $has_same_file ]; then
            if test $QUIET = "n"; then
                echo "[INFO]: has old version in ${targetdir}/$vendor_name/$1:
- Overlay Installation , please enter:[o]
- Replace directory installation , please enter: [r]
- Do not install , please enter:[n]
>>>"
        while true
                do
                    read orn
                    if [ "$orn" = n ]; then
                        return 0
                    elif [ "$orn" = o ]; then
                        break;
            elif [ "$orn" = r ]; then
                    # shellcheck disable=SC2115
                    [ -d "${targetdir}/${vendor_name}/$1/" ] && rm -rf "${targetdir}/${vendor_name}/$1"/*
            break
                    else
                        echo "[ERROR] input error, please input again!"
                    fi
                done
            fi
        fi
        log "[INFO] replace or cover ops $1 files .g....."
    fi

    log "[INFO] copy new ops $1 files ......"
    if [ -d ${targetdir}/$vendor_name/$1/ ]; then
        chmod -R +w "$targetdir/$vendor_name/$1/" >/dev/null 2>&1
    fi
    cp -rf ${sourcedir}/$vendordir/$1/* $targetdir/$vendor_name/$1/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 files failed"
        return 1
    fi
    return 0
}

handle_proto()
{   
    targetdir=$1
    if [ ! -f ${sourcedir}/$vendordir/custom.proto ]; then
        log "[INFO] no need to upgrade custom.proto files"
        return 0
    fi
    if [ ! -d ${targetdir}/$vendor_name/framework/caffe ];then
        log "[INFO] create ${targetdir}/$vendor_name/framework/caffe."
        mkdir -p ${targetdir}/$vendor_name/framework/caffe
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendor_name/framework/caffe failed"
            return 1
        fi
    else
        if [ -f ${targetdir}/$vendor_name/framework/caffe/custom.proto ]; then
            # 有老版本,判断是否要覆盖式安装
            if test $QUIET = "n"; then
                echo "[INFO] ${targetdir}/$vendor_name/framework/caffe has old version"\
                "custom.proto file. Do you want to replace? [y/n] "

                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        return 0
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        echo "[ERROR] input error, please input again!"
                    fi
                done
            fi
        log "[INFO] replace old caffe.proto files ......"
        fi
    fi
    chmod -R +w "$targetdir/$vendor_name/framework/caffe/" >/dev/null 2>&1
    cp -rf ${sourcedir}/$vendordir/custom.proto ${targetdir}/$vendor_name/framework/caffe/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new custom.proto failed"
        return 1
    fi
    log "[INFO] copy custom.proto success"
    return 0
}

check_path_exist_permission()
{
    # root
    prepare_check_path=$1
    if [ "$(id -u)" = "0" ]; then
        sh -c "test -d ${prepare_check_path} 2> /dev/null"
        if [ $? -eq 0 ]; then
            sh -c "test -w ${prepare_check_path} 2> /dev/null"
            if [ $? -eq 0 ]; then
                return 0
            else
                log "[ERROR] user do access ${prepare_check_path} failed, please check the path permission"
                return 2
            fi
        else
            log "[ERROR] user do access ${prepare_check_path} failed, please check the path valid"
            return 1
        fi
    # not root
    else
        test -d ${prepare_check_path} >> /dev/null 2>&1
        if [ $? -eq 0 ]; then
            test -w ${prepare_check_path} >> /dev/null 2>&1
            if [ $? -eq 0 ]; then
                return 0
            else
                log "[ERROR] user do access ${prepare_check_path} failed, please check the path permission"
                return 2
            fi
        else
            log "[ERROR] user do access ${prepare_check_path} failed, please check the path valid and permission"
            return 3
        fi
    fi
}

Check_Install_Path_valid()
{   
    local install_path="$1"
    # convert to absolute path
    local install_path=$(cd ${install_path} && pwd)
    # Black list: //, ...
    if echo "${install_path}" | grep -Eq '\/{2,}|\.{3,}'; then
        echo "[ERROR] Black list the install path [$1] is invalid, only characters in [a-z, A-Z, 0-9, -, _] are supported"
        return 1
    fi
    # White list
    if echo "${install_path}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        return 0
    else
        echo "[ERROR] White list the install path [$1] is invalid, only characters in [a-z, A-Z, 0-9, -, _] are supported"
        return 1
    fi
}

usage()
{
  echo "Usage:"
  echo "bash install.sh [--install-path]"
  echo ""
  echo "Options:"
  echo "    --install-path Option, set path to install custom kernels. Without this option, default dir is used."
  echo "    --help Print usage"
}

while [ $# -eq 1 ]
do
    case $1 in
    --install-path=*)
        INSTALL_PATH=$(echo $1 | cut -d"=" -f2-)
        if [ "${INSTALL_PATH}" != "/" ]; then
            INSTALL_PATH="${INSTALL_PATH%/}"
        fi
        break
    ;;
    --help)
        usage
        exit 0
    ;;
    *)
        echo "Unknown option!"
        usage
        exit 1
    ;;
    esac
done

# install-path exists and is valid, install to the specified path
if [ -n "${INSTALL_PATH}" ]; then
    last_path_status=0
    if [ ! -d ${INSTALL_PATH} ]; then
        last_path_status=1
        mkdir ${INSTALL_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${INSTALL_PATH}  failed"
            exit 1
        fi
    fi
    targetdir=${INSTALL_PATH}   
# install to the ASCEND_OPP_PATH
else
    if [[ "x${ASCEND_OPP_PATH}" == "x" ]]; then
        log "[ERROR] env ASCEND_OPP_PATH no exist"
        exit 1
    fi
    targetdir="${ASCEND_OPP_PATH}/vendors"
fi


check_path_exist_permission ${targetdir}
if [ $? -ne 0 ];then
    if [ ${last_path_status} -eq 1 ]; then
        rm -d ${targetdir} > /dev/null 2>&1
    fi
    exit 1
fi

Check_Install_Path_valid ${targetdir}
if [ $? -ne 0 ];then
    if [ ${last_path_status} -eq 1 ]; then
        rm -d ${targetdir} > /dev/null 2>&1
    fi
    exit 1
fi

log "[INFO] install package to ${targetdir}"

log "[INFO] [ops_custom] process the framework"
handle framework ${targetdir}
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] [ops_custom] process the op proto"
handle op_proto ${targetdir}
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] [ops_custom] process the op impl"
handle op_impl ${targetdir}
if [ $? -ne 0 ];then
    exit 1
fi

handle_proto ${targetdir}
if [ $? -ne 0 ];then
    exit 1
fi

# set the set_env.bash
if [ -n "${INSTALL_PATH}" ] && [ -d ${INSTALL_PATH} ]; then
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendor_name}
    bin_path="${_ASCEND_CUSTOM_OPP_PATH}/bin"
    set_env_variable="export ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:\${ASCEND_CUSTOM_OPP_PATH}"
    if [ ! -d ${bin_path} ]; then
        mkdir -p ${bin_path} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${bin_path} failed"
        fi
    fi
    cp -rf ${script_dir}/set_env.bash ${bin_path}
    if [ $? -ne 0 ]; then
        log "[ERROR] cp set_env.bash to ${bin_path} failed"
        exit 1
    fi
    sed -i "/ASCEND_CUSTOM_OPP_PATH=/c ${set_env_variable}" ${bin_path}/set_env.bash
    if [ $? -ne 0 ]; then
        log "[ERROR] write ASCEND_CUSTOM_OPP_PATH to set_env.bash failed"
        exit 1
    else
        log "[INFO] using requirements: when custom module install finished or before you run the custom module, \
        execute the command [ source ${bin_path}/set_env.bash ] to set the environment path"
    fi
elif [ "x${ASCEND_OPP_PATH}" != "x" ] && [ -d ${ASCEND_OPP_PATH} ]; then
    config_file=${targetdir}/config.ini
    if [ ! -f ${config_file} ]; then
        touch ${config_file}
        chmod 750 ${config_file}
        echo "load_priority=$vendor_name" > ${config_file}
        if [ $? -ne 0 ];then
            echo "echo load_priority failed"
            exit 1
        fi
    else
        found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
        found_vendor=$(echo $found_vendors | sed "s/$vendor_name//g" | tr ',' ' ')
        vendor=$(echo $found_vendor | tr -s ' ' ',')
        if [ "$vendor" != "" ]; then
            sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
        fi
    fi
fi

if [ $? -ne 0 ];then
    exit 1
fi

chmod -R 755 ${targetdir}/$vendor_name

log "[INFO] SUCCESS"
exit 0
