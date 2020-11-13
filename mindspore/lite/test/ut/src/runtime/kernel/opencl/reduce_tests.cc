/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/reduce.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestReduceOpenCL : public mindspore::CommonTest {
 public:
  TestReduceOpenCL() {}
};

void RunTestCaseReduce(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16,
                       int reduce_mode, bool WC = false) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->axes_[0] = 1;
  param->axes_[1] = 2;
  if (WC) {
    param->axes_[0] = 2;
    param->axes_[1] = 3;
    param->keep_dims_ = true;
  }
  param->num_axes_ = 2;
  param->mode_ = reduce_mode;
  int n = shape[0];
  int h = shape[1];
  int w = shape[2];
  int c = shape[3];
  std::vector<int> input_shape = {n, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     input_shape, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {n, c};
  if (WC) {
    out_shape = {n, h, 1, 1};
  }
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       out_shape, WC ? schema::Format_NHWC : schema::Format_NC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::ReduceOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(param), nullptr, kernel::KernelKey(), nullptr);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data, inputs[0]->ElementsNum() * dtype_size);
  pGraph->Run();

  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float16_t>(1e-3),
                  2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-3));
  }
  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test Reduce passed";
}

TEST_F(TestReduceOpenCL, ReduceMeanFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {4.5f, 5.5f, 6.5f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMean);
}

TEST_F(TestReduceOpenCL, ReduceMeanFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float16_t> output_data = {4.5f, 5.5f, 6.5f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMean);
}

TEST_F(TestReduceOpenCL, ReduceMeanLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    3.0f, 1.0f, 6.0f, 8.0f, 6.0f, 4.0f, 1.0f, 3.0f, 5.0f, 4.0f, 5.0f, 4.0f, 0.0f, 2.0f, 4.0f, 1.0f, 3.0f,
    1.0f, 6.0f, 5.0f, 4.0f, 7.0f, 0.0f, 7.0f, 1.0f, 2.0f, 5.0f, 0.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 8.0f,
    5.0f, 7.0f, 6.0f, 2.0f, 5.0f, 3.0f, 2.0f, 9.0f, 1.0f, 0.0f, 2.0f, 0.0f, 6.0f, 0.0f, 3.0f, 6.0f, 0.0f,
    7.0f, 1.0f, 0.0f, 6.0f, 3.0f, 0.0f, 1.0f, 0.0f, 5.0f, 3.0f, 8.0f, 1.0f, 9.0f, 2.0f, 2.0f, 2.0f, 7.0f,
    7.0f, 6.0f, 7.0f, 0.0f, 5.0f, 4.0f, 2.0f, 6.0f, 8.0f, 2.0f, 0.0f, 8.0f, 4.0f, 9.0f, 1.0f, 2.0f, 9.0f,
    9.0f, 6.0f, 0.0f, 8.0f, 5.0f, 2.0f, 9.0f, 3.0f, 1.0f, 9.0f, 0.0f, 4.0f, 6.0f, 0.0f, 5.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {3.971f, 4.559f, 3.294f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMean);
}

TEST_F(TestReduceOpenCL, ReduceMeanLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    3.0f, 1.0f, 6.0f, 8.0f, 6.0f, 4.0f, 1.0f, 3.0f, 5.0f, 4.0f, 5.0f, 4.0f, 0.0f, 2.0f, 4.0f, 1.0f, 3.0f,
    1.0f, 6.0f, 5.0f, 4.0f, 7.0f, 0.0f, 7.0f, 1.0f, 2.0f, 5.0f, 0.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 8.0f,
    5.0f, 7.0f, 6.0f, 2.0f, 5.0f, 3.0f, 2.0f, 9.0f, 1.0f, 0.0f, 2.0f, 0.0f, 6.0f, 0.0f, 3.0f, 6.0f, 0.0f,
    7.0f, 1.0f, 0.0f, 6.0f, 3.0f, 0.0f, 1.0f, 0.0f, 5.0f, 3.0f, 8.0f, 1.0f, 9.0f, 2.0f, 2.0f, 2.0f, 7.0f,
    7.0f, 6.0f, 7.0f, 0.0f, 5.0f, 4.0f, 2.0f, 6.0f, 8.0f, 2.0f, 0.0f, 8.0f, 4.0f, 9.0f, 1.0f, 2.0f, 9.0f,
    9.0f, 6.0f, 0.0f, 8.0f, 5.0f, 2.0f, 9.0f, 3.0f, 1.0f, 9.0f, 0.0f, 4.0f, 6.0f, 0.0f, 5.0f, 2.0f, 3.0f};
  std::vector<float16_t> output_data = {3.971f, 4.559f, 3.294f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMean);
}

TEST_F(TestReduceOpenCL, ReduceMeanWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {1.5f, 5.5f, 9.5f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMean, true);
}

TEST_F(TestReduceOpenCL, ReduceMeanWCLocalFp32) {
  int n = 1;
  int h = 5;
  int w = 17;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    6.0f, 3.0f, 6.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 7.0f, 5.0f, 7.0f, 3.0f, 0.0f, 2.0f, 9.0f, 8.0f, 3.0f, 1.0f, 6.0f,
    8.0f, 6.0f, 6.0f, 3.0f, 0.0f, 6.0f, 3.0f, 8.0f, 0.0f, 6.0f, 1.0f, 0.0f, 9.0f, 4.0f, 4.0f, 9.0f, 4.0f, 9.0f, 5.0f,
    0.0f, 1.0f, 4.0f, 6.0f, 4.0f, 0.0f, 9.0f, 3.0f, 6.0f, 6.0f, 7.0f, 1.0f, 7.0f, 8.0f, 6.0f, 0.0f, 2.0f, 6.0f, 4.0f,
    4.0f, 3.0f, 7.0f, 7.0f, 5.0f, 2.0f, 3.0f, 4.0f, 3.0f, 1.0f, 5.0f, 4.0f, 8.0f, 7.0f, 5.0f, 0.0f, 7.0f, 5.0f, 5.0f,
    0.0f, 3.0f, 4.0f, 0.0f, 6.0f, 5.0f, 4.0f, 6.0f, 2.0f, 0.0f, 8.0f, 6.0f, 4.0f, 6.0f, 3.0f, 2.0f, 6.0f, 4.0f, 8.0f,
    4.0f, 8.0f, 2.0f, 0.0f, 0.0f, 9.0f, 4.0f, 3.0f, 4.0f, 1.0f, 7.0f, 9.0f, 1.0f, 9.0f, 4.0f, 2.0f, 8.0f, 3.0f, 5.0f,
    8.0f, 7.0f, 8.0f, 8.0f, 4.0f, 8.0f, 2.0f, 8.0f, 9.0f, 4.0f, 5.0f, 0.0f, 2.0f, 1.0f, 0.0f, 8.0f, 4.0f, 7.0f, 2.0f,
    4.0f, 5.0f, 0.0f, 0.0f, 7.0f, 2.0f, 0.0f, 2.0f, 7.0f, 1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 3.0f, 7.0f, 7.0f, 3.0f,
    2.0f, 3.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 9.0f, 3.0f, 6.0f, 1.0f, 8.0f, 0.0f, 1.0f, 2.0f, 0.0f, 9.0f, 5.0f};
  std::vector<float> output_data = {4.206f, 4.441f, 4.265f, 4.706f, 3.147f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMean, true);
}

TEST_F(TestReduceOpenCL, ReduceSumFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {18.0f, 22.0f, 26.0f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSum);
}

TEST_F(TestReduceOpenCL, ReduceSumFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float16_t> output_data = {18.0f, 22.0f, 26.0f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceSum);
}

TEST_F(TestReduceOpenCL, ReduceSumLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    8.0f, 1.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 4.0f, 4.0f, 2.0f, 3.0f, 9.0f, 3.0f, 4.0f, 8.0f, 1.0f, 9.0f,
    5.0f, 2.0f, 5.0f, 6.0f, 3.0f, 8.0f, 3.0f, 7.0f, 1.0f, 3.0f, 1.0f, 9.0f, 4.0f, 0.0f, 9.0f, 7.0f, 7.0f,
    5.0f, 0.0f, 2.0f, 4.0f, 8.0f, 7.0f, 3.0f, 0.0f, 4.0f, 8.0f, 5.0f, 3.0f, 8.0f, 2.0f, 5.0f, 3.0f, 5.0f,
    9.0f, 4.0f, 3.0f, 9.0f, 7.0f, 2.0f, 4.0f, 7.0f, 0.0f, 3.0f, 9.0f, 6.0f, 6.0f, 9.0f, 2.0f, 1.0f, 0.0f,
    7.0f, 1.0f, 7.0f, 2.0f, 0.0f, 6.0f, 9.0f, 4.0f, 7.0f, 0.0f, 7.0f, 0.0f, 4.0f, 8.0f, 6.0f, 0.0f, 3.0f,
    2.0f, 1.0f, 2.0f, 9.0f, 6.0f, 2.0f, 6.0f, 2.0f, 9.0f, 4.0f, 0.0f, 1.0f, 9.0f, 7.0f, 6.0f, 9.0f, 8.0f};
  std::vector<float> output_data = {143.000f, 191.000f, 145.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSum);
}

TEST_F(TestReduceOpenCL, ReduceSumLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    8.0f, 1.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 4.0f, 4.0f, 2.0f, 3.0f, 9.0f, 3.0f, 4.0f, 8.0f, 1.0f, 9.0f,
    5.0f, 2.0f, 5.0f, 6.0f, 3.0f, 8.0f, 3.0f, 7.0f, 1.0f, 3.0f, 1.0f, 9.0f, 4.0f, 0.0f, 9.0f, 7.0f, 7.0f,
    5.0f, 0.0f, 2.0f, 4.0f, 8.0f, 7.0f, 3.0f, 0.0f, 4.0f, 8.0f, 5.0f, 3.0f, 8.0f, 2.0f, 5.0f, 3.0f, 5.0f,
    9.0f, 4.0f, 3.0f, 9.0f, 7.0f, 2.0f, 4.0f, 7.0f, 0.0f, 3.0f, 9.0f, 6.0f, 6.0f, 9.0f, 2.0f, 1.0f, 0.0f,
    7.0f, 1.0f, 7.0f, 2.0f, 0.0f, 6.0f, 9.0f, 4.0f, 7.0f, 0.0f, 7.0f, 0.0f, 4.0f, 8.0f, 6.0f, 0.0f, 3.0f,
    2.0f, 1.0f, 2.0f, 9.0f, 6.0f, 2.0f, 6.0f, 2.0f, 9.0f, 4.0f, 0.0f, 1.0f, 9.0f, 7.0f, 6.0f, 9.0f, 8.0f};
  std::vector<float16_t> output_data = {143.000f, 191.000f, 145.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceSum);
}

TEST_F(TestReduceOpenCL, ReduceSumWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {6.0f, 22.0f, 38.0f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSum, true);
}

TEST_F(TestReduceOpenCL, ReduceSumWCLocalFp32) {
  int n = 1;
  int h = 3;
  int w = 5;
  int c = 17;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    4.0f, 7.0f, 2.0f, 9.0f, 1.0f, 4.0f, 0.0f, 1.0f, 0.0f, 7.0f, 8.0f, 0.0f, 2.0f, 8.0f, 2.0f, 0.0f, 4.0f, 8.0f, 3.0f,
    9.0f, 5.0f, 9.0f, 7.0f, 0.0f, 3.0f, 3.0f, 1.0f, 1.0f, 8.0f, 6.0f, 4.0f, 7.0f, 6.0f, 5.0f, 7.0f, 8.0f, 2.0f, 0.0f,
    0.0f, 4.0f, 1.0f, 1.0f, 4.0f, 6.0f, 0.0f, 5.0f, 1.0f, 0.0f, 3.0f, 9.0f, 3.0f, 7.0f, 8.0f, 1.0f, 6.0f, 9.0f, 2.0f,
    5.0f, 7.0f, 2.0f, 9.0f, 8.0f, 0.0f, 2.0f, 0.0f, 4.0f, 3.0f, 4.0f, 3.0f, 5.0f, 3.0f, 5.0f, 2.0f, 2.0f, 1.0f, 9.0f,
    8.0f, 7.0f, 0.0f, 8.0f, 0.0f, 4.0f, 0.0f, 8.0f, 4.0f, 8.0f, 2.0f, 6.0f, 3.0f, 7.0f, 6.0f, 8.0f, 3.0f, 6.0f, 4.0f,
    8.0f, 3.0f, 8.0f, 1.0f, 0.0f, 9.0f, 6.0f, 4.0f, 9.0f, 0.0f, 6.0f, 8.0f, 6.0f, 7.0f, 8.0f, 2.0f, 3.0f, 3.0f, 7.0f,
    2.0f, 9.0f, 1.0f, 9.0f, 3.0f, 5.0f, 4.0f, 6.0f, 2.0f, 7.0f, 1.0f, 1.0f, 0.0f, 0.0f, 4.0f, 9.0f, 1.0f, 7.0f, 3.0f,
    2.0f, 1.0f, 4.0f, 6.0f, 7.0f, 9.0f, 2.0f, 2.0f, 8.0f, 3.0f, 2.0f, 4.0f, 1.0f, 7.0f, 6.0f, 8.0f, 6.0f, 9.0f, 8.0f,
    6.0f, 8.0f, 3.0f, 4.0f, 8.0f, 5.0f, 6.0f, 9.0f, 9.0f, 2.0f, 0.0f, 5.0f, 0.0f, 0.0f, 2.0f, 4.0f, 2.0f, 2.0f, 6.0f,
    9.0f, 3.0f, 6.0f, 0.0f, 5.0f, 4.0f, 3.0f, 8.0f, 6.0f, 3.0f, 2.0f, 8.0f, 9.0f, 2.0f, 7.0f, 1.0f, 2.0f, 4.0f, 9.0f,
    3.0f, 7.0f, 9.0f, 2.0f, 4.0f, 2.0f, 7.0f, 8.0f, 8.0f, 6.0f, 3.0f, 4.0f, 6.0f, 3.0f, 1.0f, 7.0f, 9.0f, 3.0f, 5.0f,
    9.0f, 7.0f, 1.0f, 8.0f, 6.0f, 1.0f, 9.0f, 2.0f, 8.0f, 2.0f, 9.0f, 8.0f, 3.0f, 2.0f, 7.0f, 8.0f, 9.0f, 3.0f, 6.0f,
    0.0f, 8.0f, 5.0f, 7.0f, 1.0f, 5.0f, 2.0f, 9.0f, 3.0f, 0.0f, 5.0f, 9.0f, 3.0f, 2.0f, 0.0f, 2.0f, 7.0f, 5.0f, 7.0f,
    4.0f, 7.0f, 0.0f, 9.0f, 8.0f, 8.0f, 8.0f, 8.0f};
  std::vector<float> output_data = {344.000f, 395.000f, 434.000f};
  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSum, true);
}

TEST_F(TestReduceOpenCL, ReduceMinFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {3.0f, -5.0f, 4.0f, 3.0f, -1.0f, 1.0f, -5.0f, -2.0f, -3.0f, 5.0f, -1.0f, 5.0f};
  std::vector<float> output_data = {-5.000f, -5.000f, -3.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMin);
}

TEST_F(TestReduceOpenCL, ReduceMinFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {3.0f, -5.0f, 4.0f, 3.0f, -1.0f, 1.0f, -5.0f, -2.0f, -3.0f, 5.0f, -1.0f, 5.0f};
  std::vector<float16_t> output_data = {-5.000f, -5.000f, -3.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMin);
}

TEST_F(TestReduceOpenCL, ReduceMinLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    2.0f,  -8.0f,  -4.0f,  -7.0f, 7.0f,   3.0f,  7.0f,  -3.0f, 2.0f,   -9.0f, -6.0f, 3.0f,   -8.0f,  1.0f,  -10.0f,
    1.0f,  -10.0f, 2.0f,   -5.0f, 6.0f,   -5.0f, 7.0f,  3.0f,  4.0f,   3.0f,  -3.0f, 5.0f,   -1.0f,  -1.0f, -6.0f,
    -4.0f, 9.0f,   5.0f,   -1.0f, 3.0f,   3.0f,  9.0f,  5.0f,  -10.0f, -1.0f, -8.0f, 9.0f,   -4.0f,  8.0f,  3.0f,
    -1.0f, -2.0f,  8.0f,   -1.0f, -7.0f,  2.0f,  4.0f,  2.0f,  4.0f,   6.0f,  -1.0f, 7.0f,   4.0f,   -3.0f, 0.0f,
    -2.0f, -1.0f,  -10.0f, -2.0f, 6.0f,   3.0f,  -4.0f, -9.0f, -5.0f,  -8.0f, 0.0f,  -7.0f,  9.0f,   2.0f,  7.0f,
    -5.0f, 8.0f,   4.0f,   5.0f,  9.0f,   -3.0f, 2.0f,  0.0f,  -4.0f,  -1.0f, -7.0f, -10.0f, -10.0f, -3.0f, 9.0f,
    -8.0f, 1.0f,   1.0f,   -5.0f, -10.0f, -1.0f, 8.0f,  -2.0f, 1.0f,   -4.0f, 1.0f,  0.0f};
  std::vector<float> output_data = {-10.000f, -10.000f, -10.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMin);
}

TEST_F(TestReduceOpenCL, ReduceMinLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    2.0f,  -8.0f,  -4.0f,  -7.0f, 7.0f,   3.0f,  7.0f,  -3.0f, 2.0f,   -9.0f, -6.0f, 3.0f,   -8.0f,  1.0f,  -10.0f,
    1.0f,  -10.0f, 2.0f,   -5.0f, 6.0f,   -5.0f, 7.0f,  3.0f,  4.0f,   3.0f,  -3.0f, 5.0f,   -1.0f,  -1.0f, -6.0f,
    -4.0f, 9.0f,   5.0f,   -1.0f, 3.0f,   3.0f,  9.0f,  5.0f,  -10.0f, -1.0f, -8.0f, 9.0f,   -4.0f,  8.0f,  3.0f,
    -1.0f, -2.0f,  8.0f,   -1.0f, -7.0f,  2.0f,  4.0f,  2.0f,  4.0f,   6.0f,  -1.0f, 7.0f,   4.0f,   -3.0f, 0.0f,
    -2.0f, -1.0f,  -10.0f, -2.0f, 6.0f,   3.0f,  -4.0f, -9.0f, -5.0f,  -8.0f, 0.0f,  -7.0f,  9.0f,   2.0f,  7.0f,
    -5.0f, 8.0f,   4.0f,   5.0f,  9.0f,   -3.0f, 2.0f,  0.0f,  -4.0f,  -1.0f, -7.0f, -10.0f, -10.0f, -3.0f, 9.0f,
    -8.0f, 1.0f,   1.0f,   -5.0f, -10.0f, -1.0f, 8.0f,  -2.0f, 1.0f,   -4.0f, 1.0f,  0.0f};
  std::vector<float16_t> output_data = {-10.000f, -10.000f, -10.000f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMin);
}

TEST_F(TestReduceOpenCL, ReduceMinWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {-0.080f, 0.481f, -0.853f, -0.838f, 0.557f, 0.255f,
                                   0.116f,  0.446f, -0.051f, -0.095f, 0.552f, 0.077f};
  std::vector<float> output_data = {-0.853f, 0.116f, -0.095f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMin, true);
}

TEST_F(TestReduceOpenCL, ReduceMinWCLocalFp32) {
  int n = 1;
  int h = 5;
  int w = 17;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    0.399f,  -0.139f, 0.238f,  0.779f,  -0.894f, 0.343f,  -0.955f, 0.593f,  0.448f,  0.816f,  0.841f,  -0.614f, 0.636f,
    0.116f,  -0.031f, -0.109f, 0.770f,  0.962f,  0.307f,  -0.170f, 0.789f,  0.197f,  0.530f,  -0.883f, 0.753f,  0.385f,
    -0.158f, 0.237f,  0.971f,  -0.781f, -0.523f, -0.547f, 0.257f,  -0.034f, 0.660f,  -0.666f, -0.379f, 0.092f,  -0.130f,
    0.369f,  0.664f,  -0.747f, -0.687f, -0.628f, -0.434f, 0.736f,  0.673f,  0.125f,  -0.854f, 0.007f,  0.038f,  0.024f,
    0.706f,  -0.806f, 0.042f,  0.532f,  -0.545f, -0.942f, 0.778f,  -0.419f, 0.931f,  -0.848f, 0.501f,  -0.415f, -0.292f,
    -0.575f, 0.192f,  -0.825f, 0.256f,  -0.227f, -0.795f, 0.319f,  0.101f,  -0.337f, 0.940f,  -0.724f, 0.453f,  -0.646f,
    -0.225f, -0.303f, 0.093f,  0.851f,  -0.467f, -0.657f, 0.980f,  0.867f,  0.606f,  0.356f,  0.982f,  -0.199f, 0.816f,
    0.984f,  -0.466f, -0.857f, -0.070f, -0.562f, 0.744f,  0.477f,  0.831f,  -0.064f, 0.891f,  -0.813f, -0.341f, 0.969f,
    0.538f,  0.233f,  -0.545f, 0.994f,  0.241f,  -0.829f, -0.272f, -0.420f, 0.607f,  0.658f,  -0.188f, 0.134f,  0.277f,
    -0.173f, 0.373f,  0.286f,  -0.805f, 0.455f,  0.461f,  0.893f,  -0.457f, 0.360f,  -0.706f, -0.848f, 0.032f,  -0.566f,
    0.014f,  0.507f,  -0.694f, -0.663f, -0.783f, 0.459f,  -0.613f, -0.496f, 0.332f,  0.829f,  -0.437f, 0.759f,  -0.061f,
    -0.400f, -0.561f, 0.471f,  -0.042f, 0.073f,  0.546f,  -0.557f, 0.602f,  0.011f,  -0.214f, 0.733f,  0.289f,  -0.847f,
    -0.637f, -0.791f, 0.519f,  0.449f,  -0.390f, -0.296f, 0.622f,  0.345f,  0.525f,  -0.205f, -0.626f, 0.089f,  -0.811f,
    0.741f};
  std::vector<float> output_data = {-0.955f, -0.942f, -0.857f, -0.848f, -0.847f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMin, true);
}

TEST_F(TestReduceOpenCL, ReduceMaxFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.123f,  0.975f, 0.092f, 0.364f,  0.033f,  -0.140f,
                                   -0.566f, 0.693f, 0.540f, -0.588f, -0.992f, -0.386f};
  std::vector<float> output_data = {0.364f, 0.975f, 0.540f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMax);
}

TEST_F(TestReduceOpenCL, ReduceMaxFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {0.123f,  0.975f, 0.092f, 0.364f,  0.033f,  -0.140f,
                                       -0.566f, 0.693f, 0.540f, -0.588f, -0.992f, -0.386f};
  std::vector<float16_t> output_data = {0.364f, 0.975f, 0.540f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMax);
}

TEST_F(TestReduceOpenCL, ReduceMaxLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    0.113f,  -0.633f, 0.603f,  0.447f,  -0.588f, 0.039f,  0.494f,  -0.379f, -0.018f, -0.317f, 0.620f,  0.460f,  0.732f,
    0.980f,  0.376f,  0.481f,  -0.371f, -0.219f, -0.496f, 0.670f,  -0.159f, 0.961f,  0.036f,  0.633f,  -0.118f, -0.300f,
    0.971f,  -0.236f, -0.095f, -0.705f, -0.495f, -0.403f, -0.131f, -0.084f, -0.339f, 0.031f,  -0.582f, 0.893f,  -0.311f,
    0.501f,  -0.623f, -0.523f, -0.177f, -0.438f, 0.626f,  0.028f,  -0.106f, 0.916f,  -0.504f, 0.678f,  0.358f,  -0.951f,
    0.741f,  -0.577f, -0.544f, -0.952f, -0.133f, 0.441f,  -0.376f, -0.246f, 0.301f,  0.025f,  -0.904f, -0.337f, 0.132f,
    -0.800f, 0.226f,  -0.135f, -0.617f, -0.871f, -0.393f, -0.195f, 0.591f,  0.034f,  -0.040f, 0.377f,  -0.106f, 0.265f,
    -0.883f, -0.678f, -0.795f, -0.094f, -0.272f, -0.954f, 0.569f,  -0.910f, -0.288f, -0.978f, 0.262f,  -0.973f, -0.750f,
    0.460f,  0.956f,  0.696f,  -0.938f, 0.537f,  0.516f,  -0.339f, -0.289f, 0.498f,  0.135f,  -0.649f};
  std::vector<float> output_data = {0.961f, 0.980f, 0.971f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMax);
}

TEST_F(TestReduceOpenCL, ReduceMaxLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    0.314f,  -0.714f, -0.736f, -0.459f, -0.819f, -0.530f, -0.275f, -0.141f, -0.797f, 0.522f,  -0.651f, 0.576f,  -0.644f,
    0.725f,  0.208f,  -0.529f, -0.776f, 0.986f,  -0.862f, -0.327f, 0.922f,  0.554f,  -0.401f, 0.972f,  -0.485f, 0.423f,
    -0.611f, -0.768f, 0.444f,  -0.678f, -0.734f, 0.572f,  0.413f,  0.612f,  -0.783f, -0.138f, -0.624f, -0.284f, 0.873f,
    -0.298f, 0.630f,  -0.463f, 0.195f,  0.196f,  0.167f,  0.227f,  -0.015f, 0.436f,  -0.898f, 0.031f,  -0.149f, -0.218f,
    0.184f,  -0.426f, 0.794f,  0.846f,  0.624f,  -0.889f, -0.336f, 0.401f,  -0.820f, -0.583f, 0.337f,  0.175f,  0.228f,
    -0.626f, -0.505f, -0.088f, 0.833f,  -0.366f, 0.392f,  0.727f,  -0.598f, -0.851f, 0.007f,  -0.707f, 0.575f,  0.243f,
    -0.372f, -0.141f, 0.679f,  -0.646f, 0.422f,  0.322f,  -0.294f, 0.831f,  0.929f,  -0.414f, -0.208f, -0.111f, 0.146f,
    -0.489f, -0.808f, -0.635f, 0.811f,  0.544f,  -0.131f, 0.707f,  0.787f,  0.603f,  -0.149f, -0.095f};
  std::vector<float16_t> output_data = {0.794f, 0.846f, 0.986f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceMax);
}

TEST_F(TestReduceOpenCL, ReduceMaxWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {0.435f,  -0.949f, 0.580f,  0.858f, -0.465f, 0.255f,
                                   -0.561f, -0.444f, -0.603f, 0.266f, 0.031f,  -0.638f};
  std::vector<float> output_data = {0.858f, 0.255f, 0.266f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMax, true);
}

TEST_F(TestReduceOpenCL, ReduceMaxWCLocalFp32) {
  int n = 1;
  int h = 5;
  int w = 17;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    0.543f,  0.620f,  0.175f,  -0.275f, -0.570f, 0.516f,  -0.401f, -0.157f, 0.460f,  -0.072f, -0.322f, 0.208f,  0.385f,
    0.919f,  -0.265f, 0.256f,  0.383f,  -0.399f, 0.183f,  0.363f,  -0.779f, -0.191f, -0.446f, 0.063f,  -0.671f, 0.823f,
    -0.049f, -0.182f, -0.409f, 0.589f,  -0.804f, -0.461f, -0.407f, -0.119f, 0.833f,  0.718f,  -0.366f, 0.993f,  0.844f,
    -0.018f, -0.203f, -0.004f, -0.610f, -0.461f, 0.938f,  -0.708f, -0.831f, -0.147f, 0.855f,  0.998f,  0.412f,  -0.393f,
    -0.706f, -0.127f, 0.845f,  -0.236f, -0.341f, 0.299f,  0.793f,  0.794f,  -0.634f, -0.663f, -0.568f, -0.428f, -0.921f,
    0.904f,  0.933f,  -0.985f, -0.760f, -0.673f, -0.080f, 0.235f,  0.539f,  -0.341f, -0.899f, 0.527f,  -0.210f, -0.151f,
    0.148f,  -0.184f, -0.103f, -0.345f, -0.772f, -0.960f, -0.282f, -0.486f, -0.986f, -0.591f, 0.702f,  0.973f,  0.269f,
    0.058f,  -0.831f, -0.677f, -0.665f, -0.403f, 0.241f,  -0.365f, 0.741f,  0.603f,  0.347f,  0.812f,  -0.515f, -0.085f,
    0.251f,  0.631f,  0.819f,  0.622f,  -0.615f, -0.122f, 0.064f,  0.445f,  -0.508f, -0.023f, -0.072f, -0.423f, 0.547f,
    -0.841f, -0.308f, 0.924f,  -0.187f, 0.601f,  0.879f,  -0.868f, 0.395f,  -0.307f, 0.977f,  -0.300f, 0.737f,  0.022f,
    0.106f,  -0.520f, -0.673f, -0.351f, 0.367f,  0.588f,  -0.223f, 0.062f,  0.870f,  -0.017f, 0.583f,  0.405f,  0.507f,
    -0.457f, 0.196f,  0.048f,  -0.173f, 0.596f,  -0.017f, -0.245f, -0.433f, -0.852f, 0.058f,  0.237f,  0.280f,  -0.129f,
    -0.224f, 0.869f,  -0.781f, -0.029f, -0.715f, 0.497f,  -0.341f, 0.230f,  -0.572f, 0.718f,  -0.408f, -0.998f, -0.752f,
    -0.701f};
  std::vector<float> output_data = {0.919f, 0.998f, 0.973f, 0.977f, 0.870f};
  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceMax, true);
}

TEST_F(TestReduceOpenCL, ReduceProdFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {4.0f, 3.0f, 1.0f, 4.0f, 1.0f, 3.0f, 1.0f, 4.0f, 2.0f, 4.0f, 4.0f, 3.0f};
  std::vector<float> output_data = {64.0f, 48.0f, 18.0f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceProd);
}

TEST_F(TestReduceOpenCL, ReduceProdFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f, 3.0f, 2.0f, 3.0f, 1.0f, 1.0f};
  std::vector<float16_t> output_data = {24.0f, 12.0f, 6.0f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceProd);
}

TEST_F(TestReduceOpenCL, ReduceProdLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    0.304f, 2.304f, 1.391f, 1.072f, 0.351f, 0.641f, 0.120f, 2.382f, 0.460f, 1.672f, 0.553f, 1.534f, 1.423f,
    0.892f, 2.900f, 1.953f, 1.745f, 1.171f, 1.717f, 1.291f, 1.572f, 2.388f, 0.154f, 0.252f, 0.794f, 0.981f,
    0.366f, 1.372f, 1.778f, 1.848f, 1.023f, 1.124f, 2.045f, 2.374f, 1.965f, 0.260f, 1.306f, 1.889f, 1.144f,
    1.816f, 2.189f, 2.215f, 1.913f, 2.577f, 2.910f, 1.712f, 0.342f, 1.349f, 0.215f, 2.717f, 1.813f, 2.764f,
    1.989f, 1.710f, 0.156f, 2.293f, 2.648f, 1.281f, 1.078f, 2.757f, 0.746f, 0.238f, 0.235f, 0.123f, 0.730f,
    1.558f, 1.798f, 0.993f, 2.479f, 1.930f, 1.687f, 1.078f, 0.600f, 0.710f, 1.926f, 0.848f, 0.984f, 0.568f,
    0.983f, 1.068f, 2.362f, 2.770f, 2.184f, 2.883f, 1.177f, 0.232f, 0.782f, 1.340f, 2.029f, 1.524f, 0.159f,
    2.892f, 1.225f, 0.638f, 2.537f, 0.813f, 0.337f, 1.871f, 0.602f, 2.387f, 1.209f, 2.886f};
  std::vector<float> output_data = {0.103f, 229.081f, 1030.031f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceProd);
}

TEST_F(TestReduceOpenCL, ReduceProdLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    2.843f, 2.398f, 0.998f, 1.164f, 1.048f, 0.880f, 2.112f, 1.354f, 2.892f, 0.755f, 2.033f, 1.140f, 1.117f,
    2.550f, 2.340f, 2.905f, 0.114f, 0.773f, 2.589f, 2.404f, 1.037f, 0.561f, 2.671f, 0.419f, 1.723f, 2.041f,
    2.888f, 2.440f, 1.668f, 0.821f, 0.918f, 1.251f, 1.141f, 2.497f, 0.408f, 2.384f, 0.457f, 2.754f, 0.624f,
    0.198f, 0.599f, 2.566f, 1.279f, 2.973f, 0.363f, 2.222f, 1.144f, 2.715f, 1.135f, 0.900f, 1.906f, 0.982f,
    2.211f, 2.113f, 0.585f, 1.766f, 1.612f, 1.796f, 0.607f, 1.121f, 1.277f, 2.600f, 1.446f, 1.467f, 1.828f,
    2.227f, 0.950f, 2.702f, 1.297f, 0.552f, 2.476f, 1.404f, 2.487f, 0.615f, 0.205f, 0.577f, 0.809f, 1.432f,
    1.668f, 2.243f, 2.711f, 2.221f, 0.183f, 2.964f, 1.174f, 0.928f, 2.703f, 0.427f, 0.410f, 1.436f, 1.427f,
    1.144f, 2.970f, 2.014f, 2.380f, 1.286f, 2.570f, 2.765f, 1.757f, 0.513f, 2.449f, 0.770f};
  std::vector<float16_t> output_data = {715.940f, 12232.266f, 46763.609f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceProd);
}

TEST_F(TestReduceOpenCL, ReduceProdWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {1.691f, 2.804f, 0.184f, 1.760f, 0.255f, 1.461f,
                                   2.751f, 2.487f, 1.304f, 0.686f, 0.702f, 0.393f};
  std::vector<float> output_data = {1.536f, 2.549f, 0.247f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceProd, true);
}

TEST_F(TestReduceOpenCL, ReduceProdWCLocalFp32) {
  int n = 1;
  int h = 5;
  int w = 17;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    1.360f, 0.615f, 0.894f, 1.357f, 0.701f, 1.430f, 1.488f, 0.701f, 0.688f, 0.869f, 1.321f, 0.836f, 1.160f, 1.460f,
    1.215f, 1.157f, 0.855f, 0.992f, 0.724f, 0.741f, 0.921f, 1.496f, 1.285f, 1.040f, 0.695f, 1.264f, 0.998f, 0.925f,
    1.170f, 1.384f, 1.413f, 0.617f, 0.743f, 1.299f, 0.998f, 1.131f, 1.491f, 1.371f, 0.808f, 1.001f, 0.602f, 0.812f,
    1.299f, 1.500f, 0.867f, 0.970f, 1.174f, 0.887f, 1.409f, 1.144f, 0.969f, 1.303f, 1.154f, 0.796f, 0.952f, 1.347f,
    0.794f, 0.601f, 1.191f, 1.310f, 0.619f, 0.961f, 0.951f, 1.395f, 0.861f, 1.177f, 1.274f, 0.701f, 0.758f, 0.635f,
    1.256f, 1.450f, 0.900f, 1.313f, 1.401f, 0.904f, 0.835f, 0.767f, 1.258f, 1.467f, 1.278f, 0.652f, 0.731f, 0.648f,
    1.308f, 1.199f, 1.485f, 1.352f, 0.639f, 1.291f, 0.924f, 0.762f, 0.791f, 1.392f, 1.328f, 1.190f, 1.458f, 1.193f,
    1.109f, 1.098f, 1.117f, 1.197f, 1.097f, 0.879f, 1.175f, 0.723f, 1.260f, 1.454f, 0.703f, 0.729f, 1.467f, 0.918f,
    0.631f, 0.750f, 1.292f, 1.208f, 0.972f, 0.621f, 0.673f, 0.710f, 1.482f, 1.092f, 1.162f, 1.432f, 0.774f, 1.132f,
    1.258f, 0.761f, 0.799f, 1.071f, 1.099f, 1.484f, 0.674f, 0.916f, 0.684f, 0.842f, 1.412f, 0.956f, 1.199f, 0.969f,
    0.957f, 1.124f, 0.937f, 0.815f, 1.308f, 1.448f, 1.059f, 1.373f, 0.804f, 1.172f, 1.387f, 0.826f, 0.783f, 0.707f,
    1.159f, 0.927f, 0.602f, 0.932f, 1.024f, 1.266f, 0.885f, 0.920f, 1.120f, 0.973f, 0.964f, 1.365f, 0.926f, 0.709f,
    1.177f, 0.615f};
  std::vector<float> output_data = {1.544f, 1.984f, 5.516f, 0.247f, 0.919f};
  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceProd, true);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {-0.081f, 0.305f,  -0.291f, 0.777f, 0.338f, 0.482f,
                                   0.959f,  -0.695f, -0.055f, 0.001f, 0.723f, -0.112f};
  std::vector<float> output_data = {1.530f, 1.213f, 0.333f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSumSquare);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {-0.730f, -0.938f, 0.236f, -0.631f, -0.058f, -0.625f,
                                       0.097f,  -0.343f, 0.120f, -0.339f, 0.003f,  -0.288f};
  std::vector<float16_t> output_data = {1.055f, 1.001f, 0.544f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceSumSquare);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareLocalFp32) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    0.025f,  -0.130f, 0.292f,  0.128f,  0.360f,  -0.181f, -0.179f, 0.469f,  0.434f,  -0.417f, -0.414f, 0.998f,  0.654f,
    -0.102f, 0.039f,  -0.822f, -0.155f, 0.113f,  0.204f,  0.615f,  0.844f,  -0.364f, 0.486f,  0.799f,  0.452f,  -0.884f,
    -0.006f, 0.888f,  -0.567f, 0.620f,  -0.365f, -0.096f, -0.300f, -0.263f, 0.945f,  -0.900f, -0.798f, -0.536f, -0.506f,
    0.148f,  -0.496f, 0.344f,  0.096f,  0.881f,  -0.848f, 0.401f,  -0.724f, 0.806f,  -0.550f, 0.377f,  0.560f,  -0.144f,
    0.439f,  0.038f,  -0.985f, 0.246f,  0.233f,  -0.864f, 0.427f,  -0.723f, 0.592f,  -0.642f, 0.376f,  0.769f,  0.020f,
    0.965f,  0.532f,  -0.448f, -0.168f, 0.502f,  0.900f,  0.468f,  0.834f,  -0.768f, -0.337f, 0.874f,  0.941f,  -0.449f,
    -0.330f, 0.605f,  0.081f,  0.804f,  -0.823f, -0.270f, 0.117f,  0.040f,  0.316f,  0.951f,  -0.920f, 0.599f,  0.855f,
    0.075f,  -0.898f, -0.298f, 0.208f,  0.899f,  0.751f,  -0.421f, 0.478f,  -0.106f, -0.031f, 0.974f};
  std::vector<float> output_data = {11.569f, 10.620f, 11.552f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSumSquare);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareLocalFp16) {
  int n = 1;
  int h = 17;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {
    0.931f,  0.611f,  0.921f,  -0.873f, 0.084f,  -0.677f, -0.366f, -0.627f, -0.359f, 0.217f,  -0.825f, -0.453f, 0.486f,
    0.675f,  -0.968f, 0.070f,  0.300f,  -0.508f, -0.423f, -0.741f, -0.390f, 0.649f,  -0.313f, -0.921f, -0.130f, -0.212f,
    -0.591f, 0.135f,  -0.556f, -0.963f, -0.509f, -0.480f, 0.694f,  -0.913f, 0.778f,  0.498f,  -0.520f, 0.271f,  0.087f,
    0.265f,  0.905f,  0.669f,  0.257f,  -0.307f, 0.789f,  0.117f,  0.468f,  0.728f,  0.372f,  -0.475f, 0.195f,  0.163f,
    0.766f,  -0.504f, 0.876f,  -0.203f, 0.636f,  -0.340f, -0.126f, 0.368f,  -0.173f, -0.149f, 0.492f,  -0.220f, 0.521f,
    -0.844f, -0.684f, -0.718f, 0.255f,  -0.148f, -0.891f, 0.577f,  -0.880f, 0.005f,  -0.904f, 0.282f,  0.473f,  -0.512f,
    -0.385f, -0.674f, 0.443f,  -0.172f, 0.224f,  0.720f,  -0.050f, 0.003f,  -0.743f, 0.025f,  0.941f,  0.107f,  0.176f,
    -0.360f, 0.975f,  -0.781f, -0.727f, 0.274f,  0.214f,  -0.330f, 0.237f,  0.967f,  0.156f,  -0.587f};
  std::vector<float16_t> output_data = {8.472f, 9.920f, 13.418f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), true, schema::ReduceMode_ReduceSumSquare);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareWCFp32) {
  int n = 1;
  int h = 3;
  int w = 2;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {-0.686f, 0.613f,  -0.701f, 0.978f, 0.632f,  0.677f,
                                   0.780f,  -0.888f, 0.147f,  0.448f, -0.100f, 0.936f};
  std::vector<float> output_data = {2.294f, 2.255f, 1.108f};

  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSumSquare, true);
}

TEST_F(TestReduceOpenCL, ReduceSumSquareWCLocalFp32) {
  int n = 1;
  int h = 5;
  int w = 17;
  int c = 2;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {
    -0.309f, -0.836f, 0.749f,  -0.820f, -0.715f, -0.770f, 0.030f,  -0.817f, 0.009f,  0.146f,  0.642f,  0.382f,  -0.085f,
    -0.268f, -0.424f, -0.957f, -0.127f, -0.852f, 0.596f,  0.340f,  -0.492f, -0.374f, -0.669f, 0.665f,  -0.664f, -0.079f,
    0.462f,  0.469f,  0.187f,  -0.730f, -0.240f, -0.446f, 0.254f,  0.284f,  0.743f,  0.297f,  0.235f,  -0.068f, 0.652f,
    -0.474f, -0.749f, -0.499f, 0.106f,  -0.988f, 0.033f,  -0.327f, -0.050f, -0.228f, -0.676f, -0.136f, -0.801f, 0.885f,
    -0.108f, -0.019f, -0.092f, 0.538f,  0.760f,  0.996f,  -0.610f, 0.125f,  0.296f,  0.861f,  0.811f,  0.948f,  -0.665f,
    0.920f,  0.669f,  0.572f,  -0.653f, -0.823f, -0.967f, -0.094f, 0.078f,  0.458f,  0.954f,  -0.357f, 0.887f,  -0.194f,
    -0.453f, -0.774f, -0.805f, -0.064f, -0.671f, -0.151f, -0.910f, 0.695f,  0.762f,  0.755f,  -0.933f, 0.277f,  -0.697f,
    0.074f,  -0.333f, 0.790f,  -0.370f, 0.264f,  -0.649f, 0.570f,  0.933f,  0.714f,  0.296f,  -0.430f, 0.634f,  0.619f,
    -0.744f, -0.898f, -0.908f, -0.800f, 0.500f,  -0.688f, 0.816f,  0.901f,  0.054f,  0.993f,  0.346f,  -0.285f, -0.926f,
    0.746f,  -0.718f, 0.708f,  -0.193f, 0.838f,  -0.869f, -0.189f, -0.195f, -0.324f, -0.498f, -0.216f, 0.632f,  -0.701f,
    0.272f,  0.550f,  0.486f,  -0.415f, 0.285f,  0.617f,  0.740f,  0.170f,  0.486f,  0.251f,  -0.165f, -0.424f, 0.705f,
    -0.802f, -0.977f, -0.449f, 0.502f,  -0.406f, 0.125f,  -0.643f, -0.324f, -0.409f, 0.218f,  0.719f,  -0.043f, -0.933f,
    -0.580f, 0.830f,  -0.091f, 0.998f,  -0.458f, 0.142f,  -0.220f, -0.440f, 0.824f,  -0.349f, 0.983f,  -0.546f, 0.085f,
    0.235f};
  std::vector<float> output_data = {9.889f, 11.926f, 13.296f, 13.537f, 10.563f};
  RunTestCaseReduce(shape, input_data.data(), output_data.data(), false, schema::ReduceMode_ReduceSumSquare, true);
}
}  // namespace mindspore
