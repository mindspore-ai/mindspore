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
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/biasadd.h"

using mindspore::kernel::BiasAddOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
class TestBiasAddOpenCL : public mindspore::CommonTest {};

void LoadDataBiasAdd(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void CompareOutBiasAdd(lite::tensor::Tensor *output_tensor, const std::string &standard_answer_file) {
  auto *output_data = reinterpret_cast<float *>(output_tensor->Data());
  size_t output_size = output_tensor->ElementsNum();
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

void printf_tensor_BiasAdd(mindspore::lite::tensor::Tensor *in_data, int size) {
  auto input_data = reinterpret_cast<float *>(in_data->Data());
  for (int i = 0; i < size; ++i) {
    printf("%f ", input_data[i]);
  }
  printf("\n");
  MS_LOG(INFO) << "Print tensor done";
}

void printf_float_BiasAdd(float *data, int num = 0) {
  float *temp = data;
  for (int i = 0; i < num; ++i) {
    std::cout << *temp << " ";
    temp++;
  }
  std::cout << std::endl;
}

TEST_F(TestBiasAddOpenCL, BiasAddFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string weight_file = "/data/local/tmp/weight_data.bin";
  std::string standard_answer_file = "/data/local/tmp/biasadd.bin";
  MS_LOG(INFO) << "BiasAdd Begin test:";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "BiasAdd init tensors.";

  std::vector<int> input_shape = {1, 9};
  std::vector<int> output_shape = {1, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error!";
    return;
  }
  auto *output_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, output_shape, schema::Format_NC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output tensor error!";
    delete input_tensor;
    return;
  }
  auto *weight_tensor = new (std::nothrow)
    lite::tensor::Tensor(data_type, std::vector<int>{input_shape[1]}, schema::Format_NHWC, tensor_type);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "new weight tensor error!";
    delete output_tensor;
    delete input_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor, weight_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  inputs[1]->MallocData(allocator);
  LoadDataBiasAdd(input_tensor->Data(), input_tensor->Size(), in_file);
  MS_LOG(INFO) << "BiasAdd==================input data================";
  printf_tensor_BiasAdd(inputs[0], input_tensor->ElementsNum());
  LoadDataBiasAdd(weight_tensor->Data(), weight_tensor->Size(), weight_file);
  MS_LOG(INFO) << "BiasAdd==================weight data================";
  printf_tensor_BiasAdd(inputs[1], weight_tensor->ElementsNum());

  auto *param = new (std::nothrow) OpParameter();
  if (param == nullptr) {
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    MS_LOG(ERROR) << "new OpParameter error!";
    return;
  }
  auto *biasadd_kernel =
    new (std::nothrow) kernel::BiasAddOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (biasadd_kernel == nullptr) {
    MS_LOG(ERROR) << "Create biasadd kernel error.";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    return;
  }

  auto ret = biasadd_kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "biasadd kernel init error.";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    delete biasadd_kernel;
    return;
  }

  MS_LOG(INFO) << "initialize sub_graph";
  std::vector<kernel::LiteKernel *> kernels{biasadd_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({input_tensor}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(ERROR) << "Create sub_graph kernel error.";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete param;
    delete biasadd_kernel;
    return;
  }
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "sub_graph init error.";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete sub_graph;
    delete param;
    delete biasadd_kernel;
    return;
  }
  MS_LOG(INFO) << "Sub graph begin running!";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "sub_graph run error.";
    delete input_tensor;
    delete output_tensor;
    delete weight_tensor;
    delete sub_graph;
    delete param;
    delete biasadd_kernel;
    return;
  }

  MS_LOG(INFO) << "BiasAdd==================output data================";
  printf_tensor_BiasAdd(outputs[0], output_tensor->ElementsNum());
  CompareOutBiasAdd(output_tensor, standard_answer_file);
  delete input_tensor;
  delete weight_tensor;
  delete output_tensor;
  delete sub_graph;
  delete param;
  delete biasadd_kernel;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}
}  // namespace mindspore
