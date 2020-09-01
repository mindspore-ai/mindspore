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
#include "mindspore/core/utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/transpose.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestTransposeOpenCL : public mindspore::CommonTest {
 public:
  TestTransposeOpenCL() {}
};

void RunTestTranspose(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = sizeof(float);
  if (enable_fp16) {
    ocl_runtime->SetFp16Enable(true);
    dtype_size = sizeof(float16_t);
  }
  auto allocator = ocl_runtime->GetAllocator();
  int h = shape[0];
  int w = shape[1];
  int c = shape[2];
  std::vector<int> input_shape = {1, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::tensor::Tensor>(
    TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), input_shape, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {1, c, h, w};
  auto tensor_out_ptr = std::make_unique<lite::tensor::Tensor>(
    TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), out_shape, schema::Format_NCHW);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{tensor_x};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};
  auto arith_kernel_ptr = std::make_unique<kernel::TransposeOpenCLKernel>(nullptr, inputs, outputs);
  auto arith_kernel = arith_kernel_ptr.get();
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
  memcpy(inputs[0]->Data(), input_data, h * w * c * dtype_size);
  pGraph->Run();

  if (enable_fp16) {
    CompareOutput(outputs[0]->Data(), output_data, h * w * c, static_cast<float16_t>(1e-3), 2e-2);
  } else {
    CompareOutput(outputs[0]->Data(), output_data, h * w * c, static_cast<float>(1e-5));
  }

  inputs[0]->SetData(nullptr);
  outputs[0]->SetData(nullptr);

  MS_LOG(INFO) << "Test TransposeFp32 passed";
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestTransposeOpenCL, TransposeFp32) {
  int h = 64;
  int w = 1;
  int c = 7360;
  std::vector<int> shape = {h, w, c};
  size_t input_size;
  std::string input_path = "./test_data/transpose/transpose_fp32_input.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input_data load error.";
    return;
  }

  size_t output_size;
  std::string output_path = "./test_data/transpose/transpose_fp32_output.bin";
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  if (correct_data == nullptr) {
    MS_LOG(ERROR) << "correct_data create error.";
    return;
  }
  RunTestTranspose(shape, input_data, correct_data, false);
}

TEST_F(TestTransposeOpenCL, TransposeFp16) {
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> shape = {h, w, c};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float16_t> output_data = {0.0f, 3.0f, 6.0f, 9.0f, 1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f, 8.0f, 11.0f};

  RunTestTranspose(shape, input_data.data(), output_data.data(), true);
}
}  // namespace mindspore
