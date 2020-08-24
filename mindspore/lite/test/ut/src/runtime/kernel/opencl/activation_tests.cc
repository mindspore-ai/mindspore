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
#include "mindspore/lite/nnacl/fp32/activation.h"
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

TEST_F(TestActivationOpenCL, ReluFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string out_file = "/data/local/tmp/relu.bin";
  MS_LOG(INFO) << "Relu Begin test!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error!";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output tensor error!";
    delete input_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  MS_LOG(INFO) << "Initialize input data";
  LoadActivationData(inputs[0]->Data(), inputs[0]->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);

  auto *param = new (std::nothrow) ActivationParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ActivationParameter fail.";
    delete input_tensor;
    delete output_tensor;
    return;
  }
  param->type_ = ActivationType_RELU;
  auto *kernel =
    new (std::nothrow) kernel::ActivationOpenClKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel:Relu create fail.";
    delete param;
    delete input_tensor;
    delete output_tensor;
    return;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete param;
    delete kernel;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Init relu fail.";
    return;
  }
  MS_LOG(INFO) << "Create kernel SubGraphOpenCLKernel.";
  std::vector<kernel::LiteKernel *> kernels{kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Kernel SubGraphOpenCLKernel create fail.";
    return;
  }

  MS_LOG(INFO) << "Initialize sub_graph.";
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub_graph error.";
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    return;
  }
  MS_LOG(INFO) << "Run SubGraphOpenCLKernel.";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    MS_LOG(ERROR) << "Run SubGraphOpenCLKernel error.";
    return;
  }

  MS_LOG(INFO) << "==================output data================";
  printf_tensor(outputs[0]);
  CompareRes(output_tensor, out_file);
  delete kernel;
  delete param;
  delete input_tensor;
  delete output_tensor;
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestActivationOpenCL, Relu6Fp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string out_file = "/data/local/tmp/relu6.bin";
  MS_LOG(INFO) << "Relu6 Begin test!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error!";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output tensor error!";
    delete input_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  MS_LOG(INFO) << "Initialize input data";
  LoadActivationData(inputs[0]->Data(), inputs[0]->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);

  auto *param = new (std::nothrow) ActivationParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ActivationParameter fail.";
    delete input_tensor;
    delete output_tensor;
    return;
  }
  param->type_ = ActivationType_RELU6;
  auto *kernel =
    new (std::nothrow) kernel::ActivationOpenClKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel:Relu6 create fail.";
    delete param;
    delete input_tensor;
    delete output_tensor;
    return;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete param;
    delete kernel;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Init relu6 fail.";
    return;
  }
  MS_LOG(INFO) << "Create kernel SubGraphOpenCLKernel.";
  std::vector<kernel::LiteKernel *> kernels{kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Kernel SubGraphOpenCLKernel create fail.";
    return;
  }

  MS_LOG(INFO) << "Initialize sub_graph.";
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub_graph error.";
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    return;
  }
  MS_LOG(INFO) << "Run SubGraphOpenCLKernel.";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    MS_LOG(ERROR) << "Run SubGraphOpenCLKernel error.";
    return;
  }

  MS_LOG(INFO) << "==================output data================";
  printf_tensor(outputs[0]);
  CompareRes(output_tensor, out_file);
  delete kernel;
  delete param;
  delete input_tensor;
  delete output_tensor;
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestActivationOpenCL, SigmoidFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string out_file = "/data/local/tmp/sigmoid.bin";
  MS_LOG(INFO) << "Sigmoid Begin test!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error!";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output tensor error!";
    delete input_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  MS_LOG(INFO) << "Initialize input data";
  LoadActivationData(inputs[0]->Data(), inputs[0]->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);

  auto *param = new (std::nothrow) ActivationParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ActivationParameter fail.";
    delete input_tensor;
    delete output_tensor;
    return;
  }
  param->type_ = ActivationType_SIGMOID;
  auto *kernel =
    new (std::nothrow) kernel::ActivationOpenClKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel:Sigmoid create fail.";
    delete param;
    delete input_tensor;
    delete output_tensor;
    return;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete param;
    delete kernel;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Init sigmoid fail.";
    return;
  }
  MS_LOG(INFO) << "Create kernel SubGraphOpenCLKernel.";
  std::vector<kernel::LiteKernel *> kernels{kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Kernel SubGraphOpenCLKernel create fail.";
    return;
  }

  MS_LOG(INFO) << "Initialize sub_graph.";
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub_graph error.";
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    return;
  }
  MS_LOG(INFO) << "Run SubGraphOpenCLKernel.";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    MS_LOG(ERROR) << "Run SubGraphOpenCLKernel error.";
    return;
  }

  MS_LOG(INFO) << "==================output data================";
  printf_tensor(outputs[0]);
  CompareRes(output_tensor, out_file);
  delete kernel;
  delete param;
  delete input_tensor;
  delete output_tensor;
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestActivationOpenCL, LeakyReluFp32_dim4) {
  std::string in_file = "/data/local/tmp/in_data.bin";
  std::string out_file = "/data/local/tmp/leaky_relu.bin";
  MS_LOG(INFO) << "Leaky relu Begin test!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Init tensors.";
  std::vector<int> input_shape = {1, 9};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  auto *input_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "new input tensor error!";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "new output tensor error!";
    delete input_tensor;
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  inputs[0]->MallocData(allocator);
  MS_LOG(INFO) << "Initialize input data";
  LoadActivationData(inputs[0]->Data(), inputs[0]->Size(), in_file);
  MS_LOG(INFO) << "==================input data================";
  printf_tensor(inputs[0]);

  auto *param = new (std::nothrow) ActivationParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ActivationParameter fail.";
    delete input_tensor;
    delete output_tensor;
    return;
  }
  param->alpha_ = 0.3;
  param->type_ = ActivationType_LEAKY_RELU;
  auto *kernel =
    new (std::nothrow) kernel::ActivationOpenClKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel:leaky relu create fail.";
    delete param;
    delete input_tensor;
    delete output_tensor;
    return;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete param;
    delete kernel;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Init leaky relu fail.";
    return;
  }
  MS_LOG(INFO) << "Create kernel SubGraphOpenCLKernel.";
  std::vector<kernel::LiteKernel *> kernels{kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    MS_LOG(ERROR) << "Kernel SubGraphOpenCLKernel create fail.";
    return;
  }

  MS_LOG(INFO) << "Initialize sub_graph.";
  ret = sub_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init sub_graph error.";
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    return;
  }
  MS_LOG(INFO) << "Run SubGraphOpenCLKernel.";
  ret = sub_graph->Run();
  if (ret != RET_OK) {
    delete kernel;
    delete param;
    delete input_tensor;
    delete output_tensor;
    delete sub_graph;
    MS_LOG(ERROR) << "Run SubGraphOpenCLKernel error.";
    return;
  }

  MS_LOG(INFO) << "==================output data================";
  printf_tensor(outputs[0]);
  CompareRes(output_tensor, out_file);
  delete kernel;
  delete param;
  delete input_tensor;
  delete output_tensor;
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}
}  // namespace mindspore
