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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/activation.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestActivationOpenCL : public mindspore::CommonTest {
 public:
  TestActivationOpenCL() {}
};

void RunTestCaseActivation(void *input_data0, const std::vector<int> &input_shape, void *output_data,
                           const std::vector<int> &out_shape, bool enable_fp16, int act_type) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Activation;
  param->type_ = act_type;
  auto tensor_x_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), input_shape);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  auto tensor_out_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), out_shape);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto op_kernel = kernel::OpenCLKernelCreator<kernel::ActivationOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(param), nullptr, kernel::KernelKey(), nullptr);
  if (op_kernel == nullptr) {
    MS_LOG(ERROR) << "op_kernel create error.";
    return;
  }
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
  memcpy(inputs[0]->MutableData(), input_data0, tensor_x->ElementsNum() * dtype_size);
  pGraph->Run();
  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, tensor_out->ElementsNum(), static_cast<float16_t>(1e-3),
                  2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, tensor_out->ElementsNum(), static_cast<float>(1e-5));
  }

  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }
  MS_LOG(INFO) << "TestActivation passed";
}

TEST_F(TestActivationOpenCL, ActivationReLUFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float> input_data = {-1.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 9.0f};
  std::vector<float> output_data = {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 9.0f};
  RunTestCaseActivation(input_data.data(), in_shape0, output_data.data(), out_shape, false,
                        schema::ActivationType_RELU);
}

TEST_F(TestActivationOpenCL, ActivationReLUFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float16_t> input_data = {-1.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 9.0f};
  std::vector<float16_t> output_data = {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 9.0f};
  RunTestCaseActivation(input_data.data(), in_shape0, output_data.data(), out_shape, true, schema::ActivationType_RELU);
}
}  // namespace mindspore
