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

namespace mindspore {
class TestMatMulOpenCL : public mindspore::CommonTest {
 public:
  TestMatMulOpenCL() {}
};

TEST_F(TestMatMulOpenCL, MatMulFp32) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  size_t input_size;
  int ci = 1280;
  int co = 1001;
  std::string input_path = "./test_data/matmul/matmul_fp32_input.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  size_t weight_size;
  std::string weight_path = "./test_data/matmul/matmul_fp32_weight.bin";
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &weight_size));

  lite::tensor::Tensor *tensor_x = new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), {1, 1, 1, ci});
  tensor_x->SetData(input_data);

  lite::tensor::Tensor *tensor_w = new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), {co, 1, 1, ci});
  tensor_w->SetData(weight_data);

  lite::tensor::Tensor *tensor_out = new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), {1, 1, 1, co});
  std::vector<lite::tensor::Tensor *> inputs{tensor_x, tensor_w};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};
  auto *arith_kernel = new kernel::MatMulOpenCLKernel(nullptr, inputs, outputs, false);
  arith_kernel->Init();

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto *pGraph = new kernel::SubGraphOpenCLKernel({tensor_x}, outputs, kernels, kernels, kernels);
  pGraph->Init();
  pGraph->Run();

  size_t output_size;
  std::string output_path = "./test_data/matmul/matmul_fp32_output.bin";
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  printf("==================output data=================\n");
  float *output_data = reinterpret_cast<float *>(tensor_out->Data());
  std::cout << std::endl;
  int size_n = co;
  size_n = size_n > 100 ? 100 : size_n;
  for (int i = 0; i < size_n; i++) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;


  // compare
  CompareOutputData(output_data, correct_data, co, 0.00001);

  delete input_data;
  delete weight_data;
  delete tensor_x;
  delete tensor_w;
  delete tensor_out;
  delete correct_data;
  MS_LOG(INFO) << "TestMatMulFp32 passed";
}
}  // namespace mindspore
