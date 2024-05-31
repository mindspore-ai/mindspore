#!/bin/bash
rm -rf ./*.log
CURDIR=$(pwd)

HOST_IP_STR=$(ifconfig -a | grep inet | grep -v 127.0.0.1 | grep -v inet6 | awk '{print $2}' | tr -d "addr:")
HOST_IPS=("$HOST_IP_STR")
HOST_IP=${HOST_IPS[-1]}

DEVICE_IP=$(cat /etc/hccn.conf | grep -E 'address_0')
DEVICE_IP=${DEVICE_IP#*=}

RANK_TABLE_FILE=${CURDIR}/rank_table_file.json
ESCLUSTER_CONFIG_PATH=${CURDIR}/escluster_config.json
echo $RANK_TABLE_FILE
echo $ESCLUSTER_CONFIG_PATH

sed -i "s/10.155.170.21/${HOST_IP}/g" ${RANK_TABLE_FILE} ${ESCLUSTER_CONFIG_PATH}
sed -i "s/192.168.100.101/${DEVICE_IP}/g" ${RANK_TABLE_FILE} ${ESCLUSTER_CONFIG_PATH}

export RANK_TABLE_FILE
export ESCLUSTER_CONFIG_PATH
export MS_DISABLE_REF_MODE=1
export JOB_ID=10087

msrun --worker_num=1 --rank_table_file=$RANK_TABLE_FILE --local_worker_num=1 python ${CURDIR}/test_es_external_api.py > es.log 2>&1 &

status=$(echo $?)
if [ "${status}" != "0" ]; then
    exit 1
fi
exit 0
