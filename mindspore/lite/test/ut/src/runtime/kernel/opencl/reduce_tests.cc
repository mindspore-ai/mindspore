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
                       int reduce_mode) {
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
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       out_shape, schema::Format_NC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel_ptr =
    std::make_unique<kernel::ReduceOpenCLKernel>(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  auto arith_kernel = arith_kernel_ptr.release();
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }
  arith_kernel->Init();

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
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-5));
  }
  for (auto t : inputs) {
    t->SetData(nullptr);
  }
  for (auto t : outputs) {
    t->SetData(nullptr);
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
}  // namespace mindspore
