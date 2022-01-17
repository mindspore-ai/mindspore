#!/bin/bash

set -e
CUR_DIR=$(cd "$(dirname $0)"; pwd)
BUILD_DIR=${CUR_DIR}/../build

export GLOG_v=2

# prepare run directory for ut
mkdir -pv ${CUR_DIR}/do_test

# prepare data for ut
cd ${CUR_DIR}/do_test
cp ${BUILD_DIR}/test/lite-test ./
cp ${BUILD_DIR}/googletest/googlemock/gtest/libgtest.so ./
ls -l *.so*
export LD_LIBRARY_PATH=./:${TENSORRT_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cp -r ${CUR_DIR}/ut/src/runtime/kernel/arm/test_data/* ./
cp -r ${CUR_DIR}/ut/tools/converter/parser/tflite/test_data/* ./
cp -r ${CUR_DIR}/ut/tools/converter/registry/test_data/* ./
# prepare data for dataset
TEST_DATA_DIR=${CUR_DIR}/../../../tests/ut/data/dataset/
cp -fr $TEST_DATA_DIR/testPK ./data

echo 'run common ut tests'
# test cases of MindData
./lite-test --gtest_filter="*MindDataTestTensorDE*"
./lite-test --gtest_filter="*MindDataTestEager*"

# test cases of Converter
## ./lite-test --gtest_filter="TestTfliteParser*"

# test cases of framework

# test cases of FP32 OP
./lite-test --gtest_filter=TestFcFp32*
./lite-test --gtest_filter=TestConv1x1Fp32*
./lite-test --gtest_filter=TestDeConvolutionFp32*

# test cases of INT8 OP
## ./lite-test --gtest_filter=TestPadInt8.*
./lite-test --gtest_filter=TestDeconvInt8.*

# test cases of GPU OpenCL
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
./lite-test --gtest_filter="ModelParserRegistryTest.TestRegistry"
./lite-test --gtest_filter="NodeParserRegistryTest.TestRegistry"
./lite-test --gtest_filter="PassRegistryTest.TestRegistry"
./lite-test --gtest_filter="TestRegistry.TestAdd"
./lite-test --gtest_filter="TestRegistryCustomOp.TestCustomAdd"

if [ -f "$BUILD_DIR/src/libmindspore-lite-train.so" ]; then
  echo 'run cxx_api ut tests'
  ./lite-test --gtest_filter="TestCxxApiLiteModel*"
  ./lite-test --gtest_filter="TestCxxApiLiteSerialization*"

  echo 'run train ut tests'
  ./lite-test --gtest_filter="TestActGradFp32*"
  ./lite-test --gtest_filter="TestSoftmaxGradFp32*"
  ./lite-test --gtest_filter="TestSoftmaxCrossEntropyFp32*"
  ./lite-test --gtest_filter="TestBiasGradFp32*"
  #./lite-test --gtest_filter="TestConvolutionGradFp32*"
  #./lite-test --gtest_filter="TestDeConvolutionGradFp32*"
fi

echo 'run inference ut tests'
./lite-test --gtest_filter="ControlFlowTest.TestMergeWhileModel"

echo 'run mindrt parallel ut test'
./lite-test --gtest_filter="MindrtParallelTest.*"
./lite-test --gtest_filter="BenchmarkTest.mindrtParallelOffline*"

echo 'user set output tensors st test'
./lite-test --gtest_filter="GraphTest.UserSetGraphOutput*"

echo 'run custom delegate st test'
./lite-test --gtest_filter="DelegateTest.CustomDelegate"

echo 'runtime pass'
./lite-test --gtest_filter="RuntimePass.*"

echo 'Runtime config file test'
./lite-test --gtest_filter="MixDataTypeTest.Config1"

echo 'run c api ut test'
./lite-test --gtest_filter="TensorCTest.*"
./lite-test --gtest_filter="ContextCTest.*"
