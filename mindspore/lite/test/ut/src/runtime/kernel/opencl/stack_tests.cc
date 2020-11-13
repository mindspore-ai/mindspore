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
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/stack.h"
namespace mindspore {
class TestStackOpenCLCI : public mindspore::CommonTest {
 public:
  TestStackOpenCLCI() {}
};

class TestStackOpenCLfp16 : public mindspore::CommonTest {
 public:
  TestStackOpenCLfp16() {}
};

TEST_F(TestStackOpenCLCI, StackFp32_8inputforCI) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 8;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
    std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8},
    std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8}, std::vector<int>{1, 1, 8}};
  std::vector<int> output_shape = {8, 1, 1, 8};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  float input_data1[] = {0.75f, 0.06f, 0.74f, 0.30f, 0.9f, 0.59f, 0.03f, 0.37f};
  float input_data2[] = {0.5f, 0.6f, 0.74f, 0.23f, 0.46f, 0.69f, 0.13f, 0.47f};
  float input_data3[] = {0.31f, 0.63f, 0.84f, 0.43f, 0.56f, 0.79f, 0.12f, 0.57f};
  float input_data4[] = {0.35f, 0.26f, 0.17f, 0.33f, 0.66f, 0.89f, 0.93f, 0.77f};
  float input_data5[] = {0.57f, 0.6f, 0.84f, 0.83f, 0.48f, 0.78f, 0.63f, 0.87f};
  float input_data6[] = {0.66f, 0.56f, 0.64f, 0.63f, 0.56f, 0.59f, 0.73f, 0.37f};
  float input_data7[] = {0.35f, 0.26f, 0.54f, 0.33f, 0.76f, 0.59f, 0.73f, 0.34f};
  float input_data8[] = {0.15f, 0.36f, 0.44f, 0.73f, 0.56f, 0.49f, 0.93f, 0.37f};
  float correctOutput[] = {0.75f, 0.06f, 0.74f, 0.30f, 0.9f,  0.59f, 0.03f, 0.37f, 0.5f,  0.6f,  0.74f, 0.23f, 0.46f,
                           0.69f, 0.13f, 0.47f, 0.31f, 0.63f, 0.84f, 0.43f, 0.56f, 0.79f, 0.12f, 0.57f, 0.35f, 0.26f,
                           0.17f, 0.33f, 0.66f, 0.89f, 0.93f, 0.77f, 0.57f, 0.6f,  0.84f, 0.83f, 0.48f, 0.78f, 0.63f,
                           0.87f, 0.66f, 0.56f, 0.64f, 0.63f, 0.56f, 0.59f, 0.73f, 0.37f, 0.35f, 0.26f, 0.54f, 0.33f,
                           0.76f, 0.59f, 0.73f, 0.34f, 0.15f, 0.36f, 0.44f, 0.73f, 0.56f, 0.49f, 0.93f, 0.37f};
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(INFO) << " new output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs{output_tensor};
  for (auto &shape : input_shapes) {
    auto input_temp = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
    inputs.push_back(input_temp);
    if (input_temp == nullptr) {
      MS_LOG(INFO) << " new input_tensor failed ";
      return;
    }
  }

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<StackParameter *>(malloc(sizeof(StackParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new StackParameter failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->axis_ = 0;
  auto *stack_kernel =
    new (std::nothrow) kernel::StackOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (stack_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::StackOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  stack_kernel->Init();
  // to do allocate memory for inputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{stack_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete stack_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data1));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data2));
  memcpy(inputs[4]->data_c(), input_data5, sizeof(input_data1));
  memcpy(inputs[5]->data_c(), input_data6, sizeof(input_data2));
  memcpy(inputs[6]->data_c(), input_data7, sizeof(input_data1));
  memcpy(inputs[7]->data_c(), input_data8, sizeof(input_data2));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.00001));
  for (auto tensor : inputs) {
    tensor->set_data(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete sub_graph;
}

TEST_F(TestStackOpenCLfp16, StackFp32_8inputaxis1) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, input2_size, input3_size, input4_size, input5_size, input6_size, input7_size, input8_size,
    output_size;
  std::string input1Ppath = "./test_data/stackfp16_input1.bin";
  std::string input2Ppath = "./test_data/stackfp16_input2.bin";
  std::string input3Ppath = "./test_data/stackfp16_input3.bin";
  std::string input4Ppath = "./test_data/stackfp16_input4.bin";
  std::string input5Ppath = "./test_data/stackfp16_input5.bin";
  std::string input6Ppath = "./test_data/stackfp16_input6.bin";
  std::string input7Ppath = "./test_data/stackfp16_input7.bin";
  std::string input8Ppath = "./test_data/stackfp16_input8.bin";
  std::string correctOutputPath = "./test_data/stackfp16_output.bin";
  auto input_data1 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto input_data3 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input3Ppath.c_str(), &input3_size));
  auto input_data4 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input4Ppath.c_str(), &input4_size));
  auto input_data5 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input5Ppath.c_str(), &input5_size));
  auto input_data6 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input6Ppath.c_str(), &input6_size));
  auto input_data7 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input7Ppath.c_str(), &input7_size));
  auto input_data8 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input8Ppath.c_str(), &input8_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 8;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
    std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18},
    std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18}, std::vector<int>{1, 17, 18}};
  std::vector<int> output_shape = {1, 8, 17, 18};
  auto data_type = kNumberTypeFloat16;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  std::vector<lite::Tensor *> inputs;
  for (auto &shape : input_shapes) {
    auto input_temp = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
    inputs.push_back(input_temp);
    if (input_temp == nullptr) {
      MS_LOG(INFO) << " new input_tensor failed ";
      return;
    }
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(INFO) << " new output_tensor failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    return;
  }
  std::vector<lite::Tensor *> outputs{output_tensor};
  MS_LOG(INFO) << " input_shapes size =: " << input_shapes.size();

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<StackParameter *>(malloc(sizeof(StackParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new StackParameter failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->axis_ = 1;
  auto *stack_kernel =
    new (std::nothrow) kernel::StackOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (stack_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::StackOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  stack_kernel->Init();
  // to  allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{stack_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete stack_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  if (inputs.size() == 8) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
    memcpy(inputs[3]->data_c(), input_data4, input4_size);
    memcpy(inputs[4]->data_c(), input_data5, input5_size);
    memcpy(inputs[5]->data_c(), input_data6, input6_size);
    memcpy(inputs[6]->data_c(), input_data7, input7_size);
    memcpy(inputs[7]->data_c(), input_data8, input8_size);
  } else {
    MS_LOG(ERROR) << " input size must be 2 or 3 or 4";
  }

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->MutableData());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001));
  for (auto tensor : inputs) {
    tensor->set_data(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete sub_graph;
}

}  // namespace mindspore
