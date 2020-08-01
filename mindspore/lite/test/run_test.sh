#!/usr/bin/env bash

set -e
CUR_DIR=$(cd "$(dirname $0)"; pwd)
BUILD_DIR=${CUR_DIR}/../build
mkdir -pv ${CUR_DIR}/do_test
cd ${CUR_DIR}/do_test
cp ${BUILD_DIR}/test/lite-test ./

./lite-test --gtest_filter="*TestHebing*"
