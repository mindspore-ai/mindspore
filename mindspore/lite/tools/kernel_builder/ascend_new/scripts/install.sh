#!/bin/bash
vendor_name=mslite
targetdir=/usr/local/Ascend/opp

script_dir=$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)
sourcedir=${script_dir}/packages
vendordir=vendors/$vendor_name
QUIET="y"

for i in "$@"
do
    echo $i
    if test $i = "--quiet"; then
        QUIET="y"
        break
    fi
done

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[runtime] [$cur_date] "$1
}

if [[ "x${ASCEND_OPP_PATH}" == "x" ]];then
    log "[ERROR] env ASCEND_OPP_PATH no exist"
    exit 1
fi

targetdir=${ASCEND_OPP_PATH}

if [ ! -d $targetdir ];then
    log "[ERROR] $targetdir no exist"
    exit 1
fi

upgrade()
{
    if [ ! -d ${sourcedir}/$vendordir/$1 ]; then
        log "[INFO] no need to upgrade ops $1 files"
        return 0
    fi

    if [ ! -d ${targetdir}/$vendordir/$1 ];then
        log "[INFO] create ${targetdir}/$vendordir/$1."
        mkdir -p ${targetdir}/$vendordir/$1
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/$1 failed"
            return 1
        fi
    else
        has_same_file=-1
        for file_a in ${sourcedir}/$vendordir/$1/*; do
            file_b=${file_a##*/};
            if [ "`ls ${targetdir}/${vendordir}/$1`" = "" ]; then
                log "[INFO] ${targetdir}/$vendordir/$1 is empty !!"
                return 1
            fi
            ls_val=`ls "${targetdir}/${vendordir}/$1"`
            grep -q $file_b <<< $ls_val;
            if [[ $? -eq 0 ]]; then
                echo -n "${file_b} "
                has_same_file=0
            fi
        done
        if [ 0 -eq $has_same_file ]; then
            if test $QUIET = "n"; then
                echo "[INFO]: has old version in ${targetdir}/$vendordir/$1, \
                you want to Overlay Installation , please enter:[o]; \
                or replace directory installation , please enter: [r]; \
                or not install , please enter:[n]."

                while true
                do
                    read orn
                    if [ "$orn" = n ]; then
                        return 0
                    elif [ "$orn" = m ]; then
                        break;
                    elif [ "$orn" = r ]; then
                        rm_target="${targetdir}/${vendordir}/$1/*"
                        rm -rf $rm_target
                        break;
                    else
                        echo "[ERROR] input error, please input again!"
                    fi
                done
            fi
        fi
        log "[INFO] replace or merge old ops $1 files .g....."
    fi

    log "copy new ops $1 files ......"
    if [ -d ${targetdir}/$vendordir/$1/ ]; then
        chmod -R +w "$targetdir/$vendordir/$1/" >/dev/null 2>&1
    fi
    cp -rf ${sourcedir}/$vendordir/$1/* $targetdir/$vendordir/$1/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 files failed"
        return 1
    fi

    return 0
}
upgrade_proto()
{
    if [ ! -f ${sourcedir}/$vendordir/custom.proto ]; then
        log "[INFO] no need to upgrade custom.proto files"
        return 0
    fi
    if [ ! -d ${targetdir}/$vendordir/framework/caffe ];then
        log "[INFO] create ${targetdir}/$vendordir/framework/caffe."
        mkdir -p ${targetdir}/$vendordir/framework/caffe
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/framework/caffe failed"
            return 1
        fi
    else
        if [ -f ${targetdir}/$vendordir/framework/caffe/custom.proto ]; then
            # 有老版本,判断是否要覆盖式安装
            if test $QUIET = "n"; then
                  echo "[INFO] ${targetdir}/$vendordir/framework/caffe has old version"\
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
        fi
        log "[INFO] replace old caffe.proto files ......"
    fi
    chmod -R +w "$targetdir/$vendordir/framework/caffe/" >/dev/null 2>&1
    cp -rf ${sourcedir}/$vendordir/custom.proto ${targetdir}/$vendordir/framework/caffe/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new custom.proto failed"
        return 1
    fi
    log "[INFO] copy custom.proto success"

    return 0
}

log "[INFO] copy uninstall sh success"

if [ ! -d ${targetdir}/vendors ];then
        log "[INFO] create ${targetdir}/vendors."
        mkdir -p ${targetdir}/vendors
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/vendors failed"
            return 1
        fi
fi
chmod u+w ${targetdir}/vendors

echo "[ops_custom]upgrade framework"
upgrade framework
if [ $? -ne 0 ];then
    exit 1
fi

echo "[ops_custom]upgrade op proto"
upgrade op_proto
if [ $? -ne 0 ];then
    exit 1
fi

echo "[ops_custom]upgrade op impl"
upgrade op_impl
if [ $? -ne 0 ];then
    exit 1
fi

upgrade_proto
if [ $? -ne 0 ];then
    exit 1
fi

config_file=${targetdir}/vendors/config.ini
if [ ! -f ${config_file} ]; then
    touch ${config_file}
    chmod 550 ${config_file}
fi

found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
found_vendor=$(echo $found_vendors | sed "s/$vendor_name//g" | tr ',' ' ')
vendor=$(echo $found_vendor | tr -s ' ' ',')
if [ "$vendor" != "" ]; then
    sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
fi

chmod u-w ${targetdir}/vendors

if [ -d ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/ ]; then
    chmod -R 440 ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/* >/dev/null 2>&1
fi
if [ -f ${targetdir}/ascend_install.info ]; then
    chmod -R 440 ${targetdir}/ascend_install.info
fi
if [ -f ${targetdir}/scene.info ]; then
    chmod -R 440 ${targetdir}/scene.info
fi
if [ -f ${targetdir}/version.info ]; then
    chmod -R 440 ${targetdir}/version.info
fi

echo "SUCCESS"
exit 0

