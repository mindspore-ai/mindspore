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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/pooling2d.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {

class TestPoolingOpenCL : public mindspore::CommonTest {};

void InitPoolingParam(PoolingParameter *param) {
  param->input_batch_ = 1;
  param->input_h_ = 2;
  param->input_w_ = 2;
  param->input_channel_ = 4;

  param->output_batch_ = 1;
  param->output_h_ = 1;
  param->output_w_ = 1;
  param->output_channel_ = 4;

  param->window_h_ = 2;
  param->window_w_ = 2;

  param->stride_h_ = 2;
  param->stride_w_ = 2;

  param->pad_u_ = 0;
  param->pad_d_ = 0;
  param->pad_l_ = 0;
  param->pad_r_ = 0;
}

void RunTestCasePooling(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16,
                        PoolMode pool_mode) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  int n = shape[0];
  int h = shape[1];
  int w = shape[2];
  int c = shape[3];
  int oh = shape[4];
  int ow = shape[5];
  auto param = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param create error.";
    return;
  }
  InitPoolingParam(param);
  param->pool_mode_ = pool_mode;
  std::vector<int> input_shape = {n, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     input_shape, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {n, oh, ow, c};
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       out_shape, schema::Format_NHWC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::PoolingOpenCLKernel>(
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
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-5));
  }

  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test AvgPool2d passed";
}

TEST_F(TestPoolingOpenCL, AvgPoolingFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 4;
  int oh = 1;
  int ow = 1;
  std::vector<int> shape = {n, h, w, c, oh, ow};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                   8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> output_data = {6.0f, 7.0f, 8.0f, 9.0f};

  RunTestCasePooling(shape, input_data.data(), output_data.data(), false, PoolMode_AvgPool);
}

TEST_F(TestPoolingOpenCL, AvgPoolingFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 4;
  int oh = 1;
  int ow = 1;
  std::vector<int> shape = {n, h, w, c, oh, ow};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                       8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float16_t> output_data = {6.0f, 7.0f, 8.0f, 9.0f};

  RunTestCasePooling(shape, input_data.data(), output_data.data(), true, PoolMode_AvgPool);
}

TEST_F(TestPoolingOpenCL, MaxPoolingFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 4;
  int oh = 1;
  int ow = 1;
  std::vector<int> shape = {n, h, w, c, oh, ow};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                   8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> output_data = {12.0f, 13.0f, 14.0f, 15.0f};

  RunTestCasePooling(shape, input_data.data(), output_data.data(), false, PoolMode_MaxPool);
}

TEST_F(TestPoolingOpenCL, MaxPoolingFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 4;
  int oh = 1;
  int ow = 1;
  std::vector<int> shape = {n, h, w, c, oh, ow};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                       8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float16_t> output_data = {12.0f, 13.0f, 14.0f, 15.0f};

  RunTestCasePooling(shape, input_data.data(), output_data.data(), true, PoolMode_MaxPool);
}
}  // namespace mindspore
