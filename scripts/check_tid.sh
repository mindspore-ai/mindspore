#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
MS_DIR=$(realpath ${SCRIPT_DIR}/..)
BUILD_DIR=${MS_DIR}/build
HASH_EXE=${BUILD_DIR}/gentid
HASH_SRC=${BUILD_DIR}/gentid.cc

mkdir -p ${BUILD_DIR}
echo "#include <iostream>"  > ${HASH_SRC}
echo "#include \"${MS_DIR}/mindspore/core/utils/hashing.h\""  >> ${HASH_SRC}
echo "int main(int argc, char *argv[0]) { std::cout << mindspore::ConstStringHash(argv[1]) << std::endl; }"  >> ${HASH_SRC}
g++ -std=c++17 -o ${HASH_EXE} ${HASH_SRC}

BASE_TID=$(${HASH_EXE} Base)
declare -A TIDMAP=( [${BASE_TID}]=Base )

grep -r MS_DECLARE_PARENT --include=*.h --include=*.cc ${MS_DIR} | while read line
do
#echo $line
if [[ "$line" =~ .*\((.*)\,(.*)\).* ]]
then
CLASS_NAME=${BASH_REMATCH[2]}_${BASH_REMATCH[1]}
TID=$(${HASH_EXE} ${CLASS_NAME})
if [ ${TIDMAP[${TID}]+_} ]; then
    echo $line
    echo Same tid $TID is used by $CLASS_NAME and ${TIDMAP[${TID}]}.
    exit 1
fi
TIDMAP[${TID}]=${CLASS_NAME}
echo ${TID} ${CLASS_NAME}
fi
done
if [ $? != 0 ];then
    echo 'Check tid failed!'
    exit 1
fi
echo 'All tids are unique, check tid ok.'
