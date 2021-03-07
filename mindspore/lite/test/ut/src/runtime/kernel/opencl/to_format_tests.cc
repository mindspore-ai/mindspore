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
#include "mindspore/lite/src/runtime/kernel/opencl/opencl_subgraph.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/to_format.h"

namespace mindspore::lite::opencl::test {
class TestToFormatOpenCL : public CommonTest {
 public:
  TestToFormatOpenCL() {}
};

TEST_F(TestToFormatOpenCL, ToFormatNHWC2NCHW) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();
  int h = 64;
  int w = 1;
  int c = 7360;
  size_t input_size;
  std::string input_path = "./test_data/transpose/transpose_fp32_input.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input_data load error.";
    return;
  }
  std::vector<int> input_shape = {1, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(kNumberTypeFloat32), input_shape, schema::Format_NHWC4);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {1, c, h, w};
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(kNumberTypeFloat32), out_shape);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel_ptr = std::make_unique<kernel::ToFormatOpenCLKernel>(nullptr, inputs, outputs, nullptr);
  auto arith_kernel = arith_kernel_ptr.get();
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }
  arith_kernel->Init();

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto pGraph_ptr = std::make_unique<kernel::OpenCLSubGraph>(inputs, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->data_c(), input_data, input_size);
  pGraph->Run();

  size_t output_size;
  std::string output_path = "./test_data/transpose/transpose_fp32_output.bin";
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  if (correct_data == nullptr) {
    MS_LOG(ERROR) << "correct_data create error.";
    return;
  }
  printf("==================output data=================\n");
  float *output_data = reinterpret_cast<float *>(tensor_out->data_c());
  std::cout << std::endl;
  int size_n = h * w * c;
  size_n = size_n > 100 ? 100 : size_n;
  for (int i = 0; i < size_n; i++) {
    std::cout << output_data[i] << " ";
    if ((i + 1) % c == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // compare
  ASSERT_EQ(0, CompareOutputData(output_data, correct_data, h * w * c, 0.00001));
  MS_LOG(INFO) << "Test TransposeFp32 passed";
}
}  // namespace mindspore::lite::opencl::test
