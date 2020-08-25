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

#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/prelu.h"
#include "mindspore/lite/nnacl/prelu_parameter.h"

using mindspore::kernel::LiteKernel;
using mindspore::kernel::PReluOpenCLKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
class TestPReluOpenCL : public mindspore::CommonTest {};

void LoadDataPRelu(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void CompareOutPRelu(lite::tensor::Tensor *output_tensor, const std::string &standard_answer_file) {
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

TEST_F(TestPReluOpenCL, PReluFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string weight_file = "/data/local/tmp/weight_data.bin";
  std::string standard_answer_file = "/data/local/tmp/caffe_prelu.bin";
  MS_LOG(INFO) << "-------------------->> Begin test PRelu!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 4, 3, 9};

  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto input_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input_tensor error!";
    return;
  }

  auto output_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output_tensor error";
    delete input_tensor;
    return;
  }

  auto weight_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, std::vector<int>{9}, schema::Format_NHWC, tensor_type);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "new weight_tensor error";
    delete input_tensor;
    delete output_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor, weight_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  // freamework to do!!! allocate memory by hand
  inputs[0]->MallocData(allocator);
  inputs[1]->MallocData(allocator);

  MS_LOG(INFO) << "initialize input data";
  LoadDataPRelu(input_tensor->Data(), input_tensor->Size(), in_file);
  LoadDataPRelu(weight_tensor->Data(), weight_tensor->Size(), weight_file);
  auto weight_data = reinterpret_cast<float *>(weight_tensor->Data());
  PrintData("Weight data", weight_data, inputs[1]->ElementsNum());
  auto *input_data = reinterpret_cast<float *>(inputs[0]->Data());
  PrintData("PRelu input data", input_data, inputs[0]->ElementsNum());
  std::cout << inputs[0]->ElementsNum() << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  auto param = new (std::nothrow) PReluParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PreluParameter error";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    return;
  }
  auto prelu_kernel =
    new (std::nothrow) kernel::PReluOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (prelu_kernel == nullptr) {
    MS_LOG(ERROR) << "new PReluOpenCLKernel error";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    return;
  }
  auto ret = prelu_kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init prelu kernel error";
    return;
  }

  MS_LOG(INFO) << "initialize sub_graph";
  std::vector<kernel::LiteKernel *> kernels{prelu_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({input_tensor}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(ERROR) << "Create kernel sub_graph error";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    delete prelu_kernel;
    return;
  }
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub graph error";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    delete prelu_kernel;
    delete sub_graph;
    return;
  }

  ret = sub_graph->Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run sub graph error";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    delete prelu_kernel;
    delete sub_graph;
    return;
  }

  MS_LOG(INFO) << "PRelu==================output data================";
  auto *output_data = reinterpret_cast<float *>(outputs[0]->Data());
  PrintData("output_data", output_data, outputs[0]->ElementsC4Num());
  CompareOutPRelu(output_tensor, standard_answer_file);
  delete input_tensor;
  delete output_tensor;
  delete weight_tensor;
  delete param;
  delete prelu_kernel;
  delete sub_graph;
}
}  // namespace mindspore
