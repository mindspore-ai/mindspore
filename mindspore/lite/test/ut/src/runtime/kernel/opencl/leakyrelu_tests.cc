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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/leaky_relu.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/leaky_relu_parameter.h"

using mindspore::kernel::LeakyReluOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;

namespace mindspore {
class TestLeakyReluOpenCL : public mindspore::CommonTest {};

void LoadDataLeakyRelu(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void CompareOutLeakyRelu(lite::tensor::Tensor *output_tensor, const std::string &standard_answer_file) {
  auto *output_data = reinterpret_cast<float *>(output_tensor->Data());
  size_t output_size = output_tensor->Size();
  auto expect_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(standard_answer_file.c_str(), &output_size));
  constexpr float atol = 0.0002;
  for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect_data[i]) > atol) {
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n\n\n", i, expect_data[i], output_data[i]);
      return;
    }
  }
  printf("compare success!\n");
  printf("compare success!\n");
  printf("compare success!\n\n\n");
}

void printf_tensor(mindspore::lite::tensor::Tensor *in_data) {
  auto input_data = reinterpret_cast<float *>(in_data->Data());
  for (int i = 0; i < in_data->ElementsNum(); ++i) {
    printf("%f ", input_data[i]);
  }
  printf("\n");
  MS_LOG(INFO) << "Print tensor done";
}

TEST_F(TestLeakyReluOpenCL, LeakyReluFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string standard_answer_file = "/data/local/tmp/out_data.bin";
  MS_LOG(INFO) << "Begin test:";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 4, 3, 8};

  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor = new lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC4, tensor_type);
  auto *output_tensor = new lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC4, tensor_type);
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  // freamework to do!!! allocate memory by hand
  inputs[0]->MallocData(allocator);

  auto param = new LeakyReluParameter();
  param->alpha = 0.3;
  auto *leakyrelu_kernel = new kernel::LeakyReluOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  leakyrelu_kernel->Init();

  MS_LOG(INFO) << "initialize sub_graph";
  std::vector<kernel::LiteKernel *> kernels{leakyrelu_kernel};
  auto *sub_graph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  sub_graph->Init();

  MS_LOG(INFO) << "initialize input data";
  LoadDataLeakyRelu(input_tensor->Data(), input_tensor->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);
  sub_graph->Run();

  MS_LOG(INFO) << "==================output data================";
  printf_tensor(outputs[0]);
  CompareOutLeakyRelu(output_tensor, standard_answer_file);
}
}  // namespace mindspore
