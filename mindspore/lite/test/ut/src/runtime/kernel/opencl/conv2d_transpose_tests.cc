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

#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/conv2d_transpose.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestConv2dTransposeOpenCL : public mindspore::CommonTest {
 public:
  TestConv2dTransposeOpenCL() {}
};

void RunTestCaseConv2dTranspose(const std::vector<int> &shape, void *input_data, void *weight_data, void *bias_data,
                                void *output_data, bool enable_fp16) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  int pad = shape[0];
  int n = shape[1];
  int h = shape[2];
  int w = shape[3];
  int kh = shape[4];
  int kw = shape[5];
  int ci = shape[6];
  int co = shape[7];
  int oh = 2 * h - 1 + 2 * (kh - 1 - pad) - kh + 1;
  int ow = 2 * w - 1 + 2 * (kw - 1 - pad) - kw + 1;
  std::vector<int> input_shape = {n, h, w, ci};
  auto tensor_x_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), input_shape);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }

  std::vector<int> weight_shape = {co, kh, kw, ci};
  auto tensor_w_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), weight_shape);
  auto tensor_w = tensor_w_ptr.get();
  if (tensor_w == nullptr) {
    MS_LOG(ERROR) << "tensor_w create error.";
    return;
  }
  tensor_w->set_data(weight_data);

  std::vector<int> bias_shape = {co};
  auto tensor_bias_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), bias_shape);
  auto tensor_bias = tensor_bias_ptr.get();
  if (tensor_bias == nullptr) {
    MS_LOG(ERROR) << "tensor_bias create error.";
    return;
  }
  tensor_bias->set_data(bias_data);

  std::vector<int> out_shape = {1, oh, ow, co};
  auto tensor_out_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), out_shape);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x, tensor_w, tensor_bias};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto opParameter = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter create error.";
    return;
  }
  opParameter->kernel_h_ = kh;
  opParameter->kernel_w_ = kw;
  opParameter->stride_h_ = 2;
  opParameter->stride_w_ = 2;
  opParameter->pad_u_ = pad;
  opParameter->pad_l_ = pad;
  opParameter->input_channel_ = ci;
  opParameter->output_channel_ = co;
  auto op_kernel = kernel::OpenCLKernelCreator<kernel::Conv2dTransposeOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(opParameter), nullptr, kernel::KernelKey(), nullptr);
  if (op_kernel == nullptr) {
    MS_LOG(ERROR) << "op_kernel create error.";
    return;
  }
  op_kernel->set_name("DeConv");

  inputs[0]->MallocData(allocator);
  std::vector<kernel::LiteKernel *> kernels{op_kernel};
  std::vector<lite::Tensor *> inputs_g{tensor_x};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs_g, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }

  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data, n * h * w * ci * dtype_size);
  pGraph->Run();
  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, n * oh * ow * co, static_cast<float16_t>(1e-3), 2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, n * oh * ow * co, static_cast<float>(1e-5));
  }

  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }
}

TEST_F(TestConv2dTransposeOpenCL, Conv2dTransposeFp32) {
  int pad = 0;
  int n = 1;
  int h = 2;
  int w = 2;
  int kh = 2;
  int kw = 2;
  int ci = 2;
  int co = 1;
  std::vector<int> shape = {pad, n, h, w, kh, kw, ci, co};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> bias_data = {0.5f};
  std::vector<float> output_data = {5.5f,  6.5f,  17.5f, 22.5f, 7.5f,  8.5f,  27.5f, 32.5f,
                                    29.5f, 38.5f, 41.5f, 54.5f, 47.5f, 56.5f, 67.5f, 80.5f};
  RunTestCaseConv2dTranspose(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), false);
}

TEST_F(TestConv2dTransposeOpenCL, Conv2dTransposeFp16) {
  int pad = 0;
  int n = 1;
  int h = 2;
  int w = 2;
  int kh = 2;
  int kw = 2;
  int ci = 2;
  int co = 1;
  std::vector<int> shape = {pad, n, h, w, kh, kw, ci, co};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float16_t> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float16_t> bias_data = {0.5f};
  std::vector<float16_t> output_data = {5.5f,  6.5f,  17.5f, 22.5f, 7.5f,  8.5f,  27.5f, 32.5f,
                                        29.5f, 38.5f, 41.5f, 54.5f, 47.5f, 56.5f, 67.5f, 80.5f};

  RunTestCaseConv2dTranspose(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), true);
}
}  // namespace mindspore
