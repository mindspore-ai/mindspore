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
#include "nnacl/pack.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/caffe_prelu.h"
#include "mindspore/lite/nnacl/prelu_parameter.h"

using mindspore::kernel::CaffePReluOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
class TestCaffePReluOpenCL : public mindspore::CommonTest {};

void LoadDataCaffePRelu(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void CompareOutCaffePRelu(lite::tensor::Tensor *output_tensor, const std::string &standard_answer_file) {
  auto *output_data = reinterpret_cast<float *>(output_tensor->Data());
  size_t output_size = output_tensor->ElementsC4Num();
  auto expect_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(standard_answer_file.c_str(), &output_size));
  constexpr float atol = 0.0002;
  for (int i = 0; i < output_tensor->ElementsC4Num(); ++i) {
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

void printf_tensor_caffeprelu(mindspore::lite::tensor::Tensor *in_data, int size) {
  auto input_data = reinterpret_cast<float *>(in_data->Data());
  for (int i = 0; i < size; ++i) {
    printf("%f ", input_data[i]);
  }
  printf("\n");
  MS_LOG(INFO) << "Print tensor done";
}

void printf_float(float *data, int num = 0) {
  float *temp = data;
  for (int i = 0; i < num; ++i) {
    std::cout << *temp << " ";
    temp++;
  }
  std::cout << std::endl;
}

TEST_F(TestCaffePReluOpenCL, CaffePReluFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string weight_file = "/data/local/tmp/weight_data.bin";
  std::string standard_answer_file = "/data/local/tmp/caffeprelu.bin";
  MS_LOG(INFO) << "CaffePRelu Begin test:";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "CaffePRelu init tensors.";

  std::vector<int> input_shape = {1, 4, 3, 9};
  std::vector<int> output_shape = {1, 4, 3, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error";
    return;
  }
  auto *output_tensor = new lite::tensor::Tensor(data_type, output_shape, schema::Format_NHWC4, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output_tensor error";
    delete input_tensor;
    return;
  }
  auto *weight_tensor = new (std::nothrow)
    lite::tensor::Tensor(data_type, std::vector<int>{input_shape[3]}, schema::Format_NHWC, tensor_type);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "new weight_tensor error";
    delete input_tensor;
    delete output_tensor;
    return;
  }

  std::vector<lite::tensor::Tensor *> inputs{input_tensor, weight_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  inputs[1]->MallocData(allocator);
  std::cout << input_tensor->Size() << std::endl;
  LoadDataCaffePRelu(input_tensor->Data(), input_tensor->Size(), in_file);
  MS_LOG(INFO) << "CaffePRelu==================input data================";
  printf_tensor_caffeprelu(inputs[0], input_tensor->ElementsNum());

  LoadDataCaffePRelu(weight_tensor->Data(), weight_tensor->Size(), weight_file);
  MS_LOG(INFO) << "CaffePRelu==================weight data================";
  printf_tensor_caffeprelu(inputs[1], weight_tensor->ElementsNum());

  auto param = new (std::nothrow) PReluParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new param error!";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    return;
  }
  param->channel_num_ = input_shape[3];
  auto *caffeprelu_kernel =
    new (std::nothrow) kernel::CaffePReluOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (caffeprelu_kernel == nullptr) {
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    MS_LOG(ERROR) << "Create caffe prelu kernel error.";
    return;
  }

  auto ret = caffeprelu_kernel->Init();
  if (ret != RET_OK) {
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete caffeprelu_kernel;
    MS_LOG(ERROR) << "caffeprelu_kernel init error.";
    return;
  }

  MS_LOG(INFO) << "initialize sub_graph";
  std::vector<kernel::LiteKernel *> kernels{caffeprelu_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({input_tensor}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete caffeprelu_kernel;
    MS_LOG(ERROR) << "Create sub_graph kernel error.";
    return;
  }
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete caffeprelu_kernel;
    delete sub_graph;
    MS_LOG(ERROR) << "sub_graph init error.";
    return;
  }
  MS_LOG(INFO) << "Sub graph begin running!";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete sub_graph;
    MS_LOG(ERROR) << "sub_graph run error.";
    return;
  }

  MS_LOG(INFO) << "CaffePRelu==================output data================";
  printf_tensor_caffeprelu(outputs[0], output_tensor->ElementsC4Num());
  CompareOutCaffePRelu(output_tensor, standard_answer_file);
  delete input_tensor;
  delete output_tensor;
  delete weight_tensor;
  delete sub_graph;
}
}  // namespace mindspore
