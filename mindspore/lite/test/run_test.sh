#!/usr/bin/env bash

set -e
CUR_DIR=$(cd "$(dirname $0)"; pwd)
BUILD_DIR=${CUR_DIR}/../build
mkdir -pv ${CUR_DIR}/do_test
cd ${CUR_DIR}/do_test
cp ${BUILD_DIR}/test/lite-test ./
cp -r ${CUR_DIR}/ut/src/runtime/kernel/arm/test_data/* ./

./lite-test --gtest_filter="*TestHebing*"

./lite-test --gtest_filter=TestFcFp32*
./lite-test --gtest_filter=TestConv1x1Fp32*
./lite-test --gtest_filter=TestStrassenFp32*
./lite-test --gtest_filter=TestDeConvolutionFp32*


./lite-test --gtest_filter=TestPadInt8*

