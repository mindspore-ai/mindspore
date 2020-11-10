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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/fullconnection.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestFullConnectionOpenCL : public mindspore::CommonTest {
 public:
  TestFullConnectionOpenCL() {}
};

void RunTestCaseFullConnection(const std::vector<int> &shape, void *input_data, void *weight_data, void *bias_data,
                               void *output_data, bool enable_fp16, int dims) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  std::vector<int> input_shape, output_shape, weight_shape, bias_shape;
  if (dims == 2) {
    int ci = shape[0];
    int co = shape[1];
    input_shape = {1, ci};
    output_shape = {1, co};
    weight_shape = {co, ci};
    bias_shape = {co};
  } else if (dims == 4) {
    int n = shape[0];
    int h = shape[1];
    int w = shape[2];
    int ci = shape[3];
    int co = shape[4];
    input_shape = {n, h, w, ci};
    output_shape = {n, co};
    weight_shape = {co, h * w * ci};
    bias_shape = {co};
  }
  auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->a_transpose_ = false;
  param->b_transpose_ = true;
  param->has_bias_ = true;
  param->act_type_ = ActType_No;
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     input_shape, dims == 2 ? schema::Format_NC : schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }

  auto tensor_w_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     weight_shape, schema::Format_NC);
  auto tensor_w = tensor_w_ptr.get();
  if (tensor_w == nullptr) {
    MS_LOG(ERROR) << "tensor_w create error.";
    return;
  }
  tensor_w->set_data(weight_data);

  auto tensor_bias_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                        bias_shape, schema::Format_NC);
  auto tensor_bias = tensor_bias_ptr.get();
  if (tensor_bias == nullptr) {
    MS_LOG(ERROR) << "tensor_w create error.";
    return;
  }
  tensor_bias->set_data(bias_data);

  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       output_shape, schema::Format_NC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x, tensor_w, tensor_bias};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto op_kernel = kernel::OpenCLKernelCreator<kernel::FullConnectionOpenCLKernel>(
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
  memcpy(inputs[0]->MutableData(), input_data, tensor_x->ElementsNum() * dtype_size);
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
  MS_LOG(INFO) << "TestFullConnection passed";
}

TEST_F(TestFullConnectionOpenCL, FullConnection2DFp32) {
  int ci = 5;
  int co = 3;
  std::vector<int> shape = {ci, co};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> bias_data = {1.0f, 1.0f, 1.0f};
  std::vector<float> output_data = {11.f, 11.f, 11.f};
  RunTestCaseFullConnection(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), false,
                            2);
}

TEST_F(TestFullConnectionOpenCL, FullConnection2DFp16) {
  int ci = 5;
  int co = 3;
  std::vector<int> shape = {ci, co};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float16_t> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float16_t> bias_data = {1.0f, 1.0f, 1.0f};
  std::vector<float16_t> output_data = {11.f, 11.f, 11.f};
  RunTestCaseFullConnection(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), true,
                            2);
}

TEST_F(TestFullConnectionOpenCL, FullConnection4DFp32) {
  int n = 1;
  int h = 2;
  int w = 1;
  int c = 4;
  int co = 2;
  std::vector<int> shape = {n, h, w, c, co};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> bias_data = {1.0f, 1.0f};
  std::vector<float> output_data = {29.f, 29.f};
  RunTestCaseFullConnection(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), false,
                            4);
}

TEST_F(TestFullConnectionOpenCL, FullConnection4DFp16) {
  int n = 1;
  int h = 2;
  int w = 1;
  int c = 4;
  int co = 2;
  std::vector<int> shape = {n, h, w, c, co};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float16_t> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float16_t> bias_data = {1.0f, 1.0f};
  std::vector<float16_t> output_data = {29.f, 29.f};
  RunTestCaseFullConnection(shape, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), true,
                            4);
}
}  // namespace mindspore
