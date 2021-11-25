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
cp ${BUILD_DIR}/googletest/googlemock/gtest/libgmock.so ./
ls -l *.so*
export LD_LIBRARY_PATH=./:${TENSORRT_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cp -r ${CUR_DIR}/ut/test_data/* ./
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
./lite-test --gtest_filter="ConvActFusionInoutTest*"
./lite-test --gtest_filter="ConvBiasFusionInoutTest*"

# test cases of framework

# test cases of FP32 OP
./lite-test --gtest_filter=TestFcFp32*
./lite-test --gtest_filter=TestConv1x1Fp32*
./lite-test --gtest_filter=TestDeConvolutionFp32*
./lite-test --gtest_filter=TestLogicalOrFp32*

# test cases of INT8 OP
## ./lite-test --gtest_filter=TestPadInt8.*
./lite-test --gtest_filter=TestDeconvInt8.*

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

echo 'runtime convert'
./lite-test --gtest_filter="RuntimeConvert.*"
./lite-test --gtest_filter="BenchmarkTest.runtimeConvert1"

echo 'Optimize Allocator'
./lite-test --gtest_filter="OptAllocator.*"

echo 'Runtime config file test'
./lite-test --gtest_filter="MixDataTypeTest.Config1"

echo 'run c api ut test'
./lite-test --gtest_filter="TensorCTest.*"
./lite-test --gtest_filter="ContextCTest.*"
