#!/usr/bin/env bash

set -e
CUR_DIR=$(cd "$(dirname $0)"; pwd)
BUILD_DIR=${CUR_DIR}/../build
mkdir -pv ${CUR_DIR}/do_test
cd ${CUR_DIR}/do_test
cp ${BUILD_DIR}/test/lite-test ./
cp -r ${CUR_DIR}/ut/src/runtime/kernel/arm/test_data/* ./
cp -r ${CUR_DIR}/ut/tools/converter/parser/tflite/test_data/* ./
## prepare data for dataset
TEST_DATA_DIR=${CUR_DIR}/../../../tests/ut/data/dataset/
cp -fr $TEST_DATA_DIR/testPK ./data

./lite-test --gtest_filter="*MindDataTestTensorDE*"
./lite-test --gtest_filter="*MindDataTestEager*"

./lite-test --gtest_filter="TestTfliteParser*"

./lite-test --gtest_filter="*TestHebing*"

./lite-test --gtest_filter=TestFcFp32*
./lite-test --gtest_filter=TestConv1x1Fp32*
./lite-test --gtest_filter=TestStrassenFp32*
./lite-test --gtest_filter=TestDeConvolutionFp32*

./lite-test --gtest_filter=TestPadInt8.*
./lite-test --gtest_filter=TestDeconvInt8.*

./lite-test --gtest_filter="TestTfliteParser*"

# for GPU OpenCL
./lite-test --gtest_filter="TestConvolutionOpenCL.simple_test*"
./lite-test --gtest_filter="TestArithmeticSelfOpenCLCI.ArithmeticSelfRound*"
./lite-test --gtest_filter="TestConcatOpenCLCI.ConcatFp32_2inputforCI*"
./lite-test --gtest_filter="TestSliceOpenCLfp32.Slicefp32CI*"
./lite-test --gtest_filter="TestBatchnormOpenCLCI.Batchnormfp32CI*"
./lite-test --gtest_filter="TestAvgPoolingOpenCL*"
./lite-test --gtest_filter="TestConv2dTransposeOpenCL*"
./lite-test --gtest_filter="TestMatMulOpenCL*"
./lite-test --gtest_filter="TestMaxPoolingOpenCL*"
./lite-test --gtest_filter="TestReduceOpenCL*"
./lite-test --gtest_filter="TestReshapeOpenCL*"
./lite-test --gtest_filter="TestSoftmaxOpenCL*"
./lite-test --gtest_filter="TestTransposeOpenCL*"
./lite-test --gtest_filter="TestArithmeticOpenCL*"
./lite-test --gtest_filter="TestScaleOpenCL*"
./lite-test --gtest_filter="TestFullConnectionOpenCL*"
./lite-test --gtest_filter="TestResizeOpenCL*"
./lite-test --gtest_filter="TestSwishOpenCLCI.Fp32CI"
