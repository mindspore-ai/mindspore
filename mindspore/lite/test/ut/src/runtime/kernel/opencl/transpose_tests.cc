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

namespace mindspore {
class TestTransposeOpenCL : public mindspore::CommonTest {
 public:
  TestTransposeOpenCL() {}
};

TEST_F(TestTransposeOpenCL, TransposeFp32) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();
  int h = 64;
  int w = 1;
  int c = 7360;
  size_t input_size;
  std::string input_path = "./test_data/transpose/transpose_fp32_input.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  lite::tensor::Tensor *tensor_x =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), {1, h, w, c}, schema::Format_NHWC4);

  lite::tensor::Tensor *tensor_out = new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), {1, c, h, w});
  std::vector<lite::tensor::Tensor *> inputs{tensor_x};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};
  auto *arith_kernel = new kernel::TransposeOpenCLKernel(nullptr, inputs, outputs);
  arith_kernel->Init();

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  pGraph->Init();
  memcpy(inputs[0]->Data(), input_data, input_size);
  pGraph->Run();

  size_t output_size;
  std::string output_path = "./test_data/transpose/transpose_fp32_output.bin";
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  printf("==================output data=================\n");
  float *output_data = reinterpret_cast<float *>(tensor_out->Data());
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
  CompareOutputData(output_data, correct_data, h * w * c, 0.00001);
  MS_LOG(INFO) << "TestMatMulFp32 passed";
}
}  // namespace mindspore
