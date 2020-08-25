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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/matmul.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestMatMulOpenCL : public mindspore::CommonTest {
 public:
  TestMatMulOpenCL() {}
};

void RunTestCaseMatMul(const std::vector<int> shape, const std::vector<std::string> file_path, bool fp16) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  if (fp16) {
    ocl_runtime->SetFp16Enable(true);
  }
  auto allocator = ocl_runtime->GetAllocator();
  size_t input_size;
  int ci = shape[0];
  int co = shape[1];
  std::string input_path = file_path[0];
  auto input_data = mindspore::lite::ReadFile(input_path.c_str(), &input_size);
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input_data load error.";
    return;
  }
  size_t weight_size;
  std::string weight_path = file_path[1];
  auto weight_data = mindspore::lite::ReadFile(weight_path.c_str(), &weight_size);
  if (weight_data == nullptr) {
    MS_LOG(ERROR) << "weight_data load error.";
    return;
  }
  std::vector<int> input_shape = {1, ci};
  auto tensor_x_ptr = std::make_unique<lite::tensor::Tensor>(TypeId(fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                             input_shape, schema::Format_NC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }

  std::vector<int> w_shape = {co, ci};
  auto tensor_w_ptr =
    std::make_unique<lite::tensor::Tensor>(TypeId(fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), w_shape);
  auto tensor_w = tensor_w_ptr.get();
  if (tensor_w == nullptr) {
    MS_LOG(ERROR) << "tensor_w create error.";
    return;
  }
  tensor_w->SetData(weight_data);

  std::vector<int> out_shape = {1, co};
  auto tensor_out_ptr = std::make_unique<lite::tensor::Tensor>(TypeId(fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                               out_shape, schema::Format_NC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{tensor_x, tensor_w};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};
  auto op_kernel_ptr = std::make_unique<kernel::MatMulOpenCLKernel>(nullptr, inputs, outputs, false);
  auto op_kernel = op_kernel_ptr.get();
  if (op_kernel == nullptr) {
    MS_LOG(ERROR) << "op_kernel create error.";
    return;
  }
  op_kernel->Init();
  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{op_kernel};

  std::vector<lite::tensor::Tensor *> inputs_g{tensor_x};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs_g, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->Data(), input_data, input_size);
  pGraph->Run();
  if (fp16) {
    CompareOutput(tensor_out, file_path[2], static_cast<float16_t>(1e-3), 2e-2);
  } else {
    CompareOutput(tensor_out, file_path[2], static_cast<float>(1e-5));
  }

  tensor_x->SetData(nullptr);
  tensor_out->SetData(nullptr);
  MS_LOG(INFO) << "TestMatMulFp32 passed";
}

TEST_F(TestMatMulOpenCL, MatMulFp32) {
  int ci = 1280;
  int co = 1001;
  std::vector<int> shape = {ci, co};
  std::vector<std::string> file_path = {"./test_data/matmul/matmul_fp32_input.bin",
                                        "./test_data/matmul/matmul_fp32_weight.bin",
                                        "./test_data/matmul/matmul_fp32_output.bin"};
  RunTestCaseMatMul(shape, file_path, false);
}

TEST_F(TestMatMulOpenCL, MatMulFp16) {
  int ci = 1280;
  int co = 1001;
  std::vector<int> shape = {ci, co};
  std::vector<std::string> file_path = {"./test_data/matmul/matmul_fp16_input.bin",
                                        "./test_data/matmul/matmul_fp16_weight.bin",
                                        "./test_data/matmul/matmul_fp16_output.bin"};
  RunTestCaseMatMul(shape, file_path, true);
}
}  // namespace mindspore
