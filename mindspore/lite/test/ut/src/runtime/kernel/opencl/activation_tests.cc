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
#include "mindspore/lite/src/runtime/opencl/opencl_allocator.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/activation.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/activation.h"

using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore {
class TestActivationOpenCL : public mindspore::CommonTest {};

void LoadActivationData(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void CompareRes(lite::tensor::Tensor *output_tensor, const std::string &standard_answer_file) {
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

kernel::ActivationOpenClKernel *create_kernel(lite::opencl::OpenCLAllocator *allocator,
                                              const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs, std::string test_name,
                                              int type, std::string in_file, float alpha = 0.2) {
  auto *param = new (std::nothrow) ActivationParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ActivationParameter fail.";
    return nullptr;
  }
  memcpy(param->op_parameter_.name_, test_name.c_str(), test_name.size());
  param->alpha_ = alpha;
  param->type_ = type;
  auto *kernel =
    new (std::nothrow) kernel::ActivationOpenClKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel:" << test_name << " create fail.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init " << test_name << " fail.";
    return nullptr;
  }
  MS_LOG(INFO) << "Initialize input data";
  LoadActivationData(inputs[0]->Data(), inputs[0]->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);
  return kernel;
}

int RunSubGraphOpenCLKernel(const std::vector<lite::tensor::Tensor *> &inputs,
                            const std::vector<lite::tensor::Tensor *> &outputs,
                            kernel::ActivationOpenClKernel *kernel) {
  MS_LOG(INFO) << "Create kernel SubGraphOpenCLKernel.";
  std::vector<kernel::LiteKernel *> kernels{kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(ERROR) << "Kernel SubGraphOpenCLKernel create fail.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "Initialize sub_graph.";
  auto ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub_graph error.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "Run SubGraphOpenCLKernel.";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run SubGraphOpenCLKernel error.";
    return RET_ERROR;
  }
  return RET_OK;
}

TEST_F(TestActivationOpenCL, LeakyReluFp32_dim4) {
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

  std::map<std::string, int> Test_Activation_Type;
  std::map<std::string, std::string> Test_Res_File;
  Test_Activation_Type["Relu"] = ActivationType_RELU;
  Test_Activation_Type["Leaky_Relu"] = ActivationType_LEAKY_RELU;
  Test_Activation_Type["Relu6"] = ActivationType_RELU6;
  Test_Activation_Type["Sigmoid"] = ActivationType_SIGMOID;
  Test_Res_File["Leaky_Relu"] = "/data/local/tmp/leaky_relu.bin";
  Test_Res_File["Relu"] = "/data/local/tmp/relu.bin";
  Test_Res_File["Relu6"] = "/data/local/tmp/relu6.bin";
  Test_Res_File["Sigmoid"] = "/data/local/tmp/sigmoid.bin";
  std::string in_file = "/data/local/tmp/in_data.bin";

  std::map<std::string, int>::iterator it = Test_Activation_Type.begin();
  while (it != Test_Activation_Type.end()) {
    auto kernel = create_kernel(allocator, inputs, outputs, it->first, it->second, in_file, 0.3);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel:" << it->first << " error.";
      return;
    }

    auto ret = RunSubGraphOpenCLKernel(inputs, outputs, kernel);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << it->first << " RunSubGraphOpenCLKernel error.";
      return;
    }
    MS_LOG(INFO) << "==================output data================";
    printf_tensor(outputs[0]);
    CompareRes(output_tensor, Test_Res_File[it->first]);
    delete kernel;
    it++;
  }

  delete input_tensor;
  delete output_tensor;
  return;
}
}  // namespace mindspore
