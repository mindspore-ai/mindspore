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
#include "src/runtime/kernel/opencl/utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/gather.h"

namespace mindspore {
class TestGatherOpenCL : public mindspore::CommonTest {
 public:
  TestGatherOpenCL() {}
};

template <typename T>
void test_main_gather(void *input_data, void *correct_data, const std::vector<int> &input_shape,
                      const std::vector<int> &indices, GatherParameter *param, TypeId data_type,
                      schema::Format format) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_wrp = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = ocl_wrp.GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  std::vector<int> indices_shape = {static_cast<int>(indices.size())};
  std::vector<int> output_shape = input_shape;
  output_shape[param->axis_] = indices.size();

  auto tensor_a = lite::Tensor(TypeId(data_type), input_shape, format);
  auto tensor_b = lite::Tensor(kNumberTypeInt32, indices_shape, schema::Format_NC);
  auto tensor_c = lite::Tensor(TypeId(data_type), output_shape, format);
  std::vector<lite::Tensor *> inputs{&tensor_a, &tensor_b};
  std::vector<lite::Tensor *> outputs{&tensor_c};
  size_t input_size = tensor_a.Size();

  auto *pkernel =
    new (std::nothrow) kernel::GatherOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (pkernel == nullptr) {
    MS_LOG(INFO) << "new GatherOpenCLKernel failed ";
    return;
  }
  pkernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{pkernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&tensor_a}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete pkernel;
    MS_LOG(INFO) << " new SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);
  auto input1_tensor = reinterpret_cast<int *>(inputs[1]->data_c());
  for (int i = 0; i < inputs[1]->ElementsNum(); ++i) {
    input1_tensor[i] = indices.at(i);
  }
  sub_graph->Run();

  std::cout << "==================output data================" << std::endl;
  auto *output_data = reinterpret_cast<T *>(outputs[0]->data_c());
  for (size_t i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "==================expected data================" << std::endl;
  for (size_t i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << static_cast<T *>(correct_data)[i] << " ";
  }
  std::cout << std::endl;
  CommonTest::CompareOutputData(output_data, static_cast<T *>(correct_data), outputs[0]->ElementsNum(), 0.0001);
}
TEST_F(TestGatherOpenCL, Axis0Fp16) {
  std::vector<int> input_shape{5, 10, 10, 5};
  std::vector<int> indices{1, 0, 3, 4};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
  param->axis_ = 0;
  size_t input_size, output_size;
  std::string inputPpath = "./test_data/gatherfp16_input.bin";
  std::string correctOutputPath = "./test_data/gatherfp16_output.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(inputPpath.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  if (param == nullptr) {
    return;
  }
  TypeId data_type = kNumberTypeFloat16;
  schema::Format format = schema::Format_NHWC;
  test_main_gather<float16_t>(input_data, correct_data, input_shape, indices, param, data_type, format);
}

TEST_F(TestGatherOpenCL, Axis0Fp32) {
  std::vector<int> input_shape{5, 10, 10, 5};
  std::vector<int> indices{1, 2, 3, 4};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
  param->axis_ = 0;
  size_t input_size, output_size;
  std::string inputPpath = "./test_data/gatherfp32_input.bin";
  std::string correctOutputPath = "./test_data/gatherfp32_output.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(inputPpath.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  if (param == nullptr) {
    return;
  }
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_gather<float>(input_data, correct_data, input_shape, indices, param, data_type, format);
}

TEST_F(TestGatherOpenCL, Axis1Fp32) {
  std::vector<int> input_shape{1, 5, 4, 4};
  std::vector<int> indices{1, 3};
  GatherParameter *param = reinterpret_cast<GatherParameter *>(malloc(sizeof(GatherParameter)));
  param->axis_ = 1;
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  float correct_data[] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                          48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  if (param == nullptr) {
    return;
  }
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_gather<float>(input_data, correct_data, input_shape, indices, param, data_type, format);
}

TEST_F(TestGatherOpenCL, Axis2Fp32) {
  std::vector<int> input_shape{1, 5, 4, 4};
  std::vector<int> indices{1, 3};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
  param->axis_ = 2;
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  float correct_data[] = {4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                          44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71, 76, 77, 78, 79};
  if (param == nullptr) {
    return;
  }
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_gather<float>(input_data, correct_data, input_shape, indices, param, data_type, format);
}

TEST_F(TestGatherOpenCL, Axis3Fp32) {
  std::vector<int> input_shape{1, 5, 4, 4};
  std::vector<int> indices{1, 3};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
  param->axis_ = 3;
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  float correct_data[] = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                          41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79};
  if (param == nullptr) {
    return;
  }
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_gather<float>(input_data, correct_data, input_shape, indices, param, data_type, format);
}
}  // namespace mindspore
