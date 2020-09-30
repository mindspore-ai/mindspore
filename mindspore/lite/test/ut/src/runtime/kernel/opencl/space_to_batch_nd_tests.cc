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
#include "src/runtime/kernel/opencl/utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/space_to_batch_nd.h"

namespace mindspore {
class TestSpaceToBatchNDOpenCL : public mindspore::CommonTest {
 public:
  TestSpaceToBatchNDOpenCL() {}
};
template <typename T>
void test_main_space_to_batch_nd(void *input_data, void *correct_data, const std::vector<int> &input_shape,
                                 SpaceToBatchParameter *param, TypeId data_type, schema::Format format) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime_wrap = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = ocl_runtime_wrap.GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  std::vector<int> output_shape = input_shape;
  output_shape[0] = input_shape[0] * param->block_sizes_[0] * param->block_sizes_[1];
  output_shape[1] = (input_shape[1] + param->paddings_[0] + param->paddings_[1]) / param->block_sizes_[0];
  output_shape[2] = (input_shape[2] + +param->paddings_[2] + param->paddings_[3]) / param->block_sizes_[1];

  auto tensor_a = lite::Tensor(TypeId(data_type), input_shape, format);
  auto tensor_c = lite::Tensor(TypeId(data_type), output_shape, format);
  std::vector<lite::Tensor *> inputs{&tensor_a};
  std::vector<lite::Tensor *> outputs{&tensor_c};
  size_t input_size = tensor_a.Size();

  auto *pkernel =
    new (std::nothrow) kernel::SpaceToBatchNDOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (pkernel == nullptr) {
    MS_LOG(INFO) << "new SpaceToBatchNDOpenCLKernel failed ";
    return;
  }
  pkernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{pkernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete pkernel;
    MS_LOG(INFO) << " new SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->MutableData());
  memcpy(input_ptr, input_data, input_size);
  std::cout << "==================input data================" << std::endl;
  for (auto i = 0; i < inputs[0]->ElementsNum(); ++i) {
    std::cout << input_ptr[i] << ", ";
  }
  std::cout << std::endl;

  sub_graph->Run();

  auto *output_data = reinterpret_cast<T *>(outputs[0]->MutableData());
  std::cout << "==================output data================" << std::endl;
  for (auto i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "==================correct data================" << std::endl;
  for (auto i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << static_cast<T *>(correct_data)[i] << ", ";
  }
  std::cout << std::endl;
  CommonTest::CompareOutputData<T>(output_data, static_cast<T *>(correct_data), outputs[0]->ElementsNum(), 0.0001);
  delete sub_graph;
}
TEST_F(TestSpaceToBatchNDOpenCL, NHWC4H2W2Pad2222) {
  std::vector<int> input_shape{1, 6, 6, 4};
  SpaceToBatchParameter *param = std::make_unique<SpaceToBatchParameter>().release();
  if (param == nullptr) {
    return;
  }
  param->block_sizes_[0] = 2;
  param->block_sizes_[1] = 2;
  param->paddings_[0] = 2;
  param->paddings_[1] = 2;
  param->paddings_[2] = 2;
  param->paddings_[3] = 2;
  float input_data[] = {172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140,
                        58,  193, 230, 39,  87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197,
                        254, 79,  175, 192, 82,  99,  216, 177, 243, 29,  147, 147, 142, 167, 32,  193, 9,   185,
                        127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,  34,  128, 128, 164, 53,  133,
                        38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,  35,  102,
                        119, 11,  174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,
                        127, 244, 131, 204, 100, 180, 232, 78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,
                        49,  69,  169, 163, 192, 95,  197, 94,  0,   113, 178, 36,  162, 48,  93,  131, 98,  42};
  float correct_data[] = {
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   172, 47, 117, 192, 9,   211, 21,  242, 88,  140, 58,  193, 0,   0,   0,   0,   0,   0,   0,   0,   142, 167,
    32,  193, 31, 202, 244, 151, 183, 28,  34,  128, 0,   0,   0,   0,   0,   0,   0,   0,   142, 99,  53,  140, 68,
    6,   196, 47, 100, 180, 232, 78,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   67,  251, 195, 103, 36,  87,  70,  216, 230, 39,  87,  174, 0,   0,
    0,   0,   0,  0,   0,   0,   9,   185, 127, 32,  163, 254, 203, 114, 128, 164, 53,  133, 0,   0,   0,   0,   0,
    0,   0,   0,  121, 170, 84,  203, 127, 244, 131, 204, 143, 148, 227, 186, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   88,  81,  165, 25,  115, 208,
    243, 197, 82, 99,  216, 177, 0,   0,   0,   0,   0,   0,   0,   0,   38,  232, 244, 17,  186, 31,  120, 1,   35,
    102, 119, 11, 0,   0,   0,   0,   0,   0,   0,   0,   23,  207, 141, 117, 169, 163, 192, 95,  178, 36,  162, 48,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   77, 72,  9,   148, 254, 79,  175, 192, 243, 29,  147, 147, 0,   0,   0,   0,   0,   0,   0,   0,   79,
    132, 105, 42, 65,  231, 169, 57,  174, 82,  91,  128, 0,   0,   0,   0,   0,   0,   0,   0,   85,  48,  49,  69,
    197, 94,  0,  113, 93,  131, 98,  42,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0};
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_space_to_batch_nd<float>(input_data, correct_data, input_shape, param, data_type, format);
}
TEST_F(TestSpaceToBatchNDOpenCL, NC4HW4H2W2Pad2222) {
  std::vector<int> input_shape{1, 6, 6, 4};
  SpaceToBatchParameter *param = std::make_unique<SpaceToBatchParameter>().release();
  if (param == nullptr) {
    return;
  }
  param->block_sizes_[0] = 2;
  param->block_sizes_[1] = 2;
  param->paddings_[0] = 2;
  param->paddings_[1] = 2;
  param->paddings_[2] = 2;
  param->paddings_[3] = 2;
  float input_data[] = {172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140,
                        58,  193, 230, 39,  87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197,
                        254, 79,  175, 192, 82,  99,  216, 177, 243, 29,  147, 147, 142, 167, 32,  193, 9,   185,
                        127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,  34,  128, 128, 164, 53,  133,
                        38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,  35,  102,
                        119, 11,  174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,
                        127, 244, 131, 204, 100, 180, 232, 78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,
                        49,  69,  169, 163, 192, 95,  197, 94,  0,   113, 178, 36,  162, 48,  93,  131, 98,  42};
  float correct_data[] = {
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   172, 47, 117, 192, 9,   211, 21,  242, 88,  140, 58,  193, 0,   0,   0,   0,   0,   0,   0,   0,   142, 167,
    32,  193, 31, 202, 244, 151, 183, 28,  34,  128, 0,   0,   0,   0,   0,   0,   0,   0,   142, 99,  53,  140, 68,
    6,   196, 47, 100, 180, 232, 78,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   67,  251, 195, 103, 36,  87,  70,  216, 230, 39,  87,  174, 0,   0,
    0,   0,   0,  0,   0,   0,   9,   185, 127, 32,  163, 254, 203, 114, 128, 164, 53,  133, 0,   0,   0,   0,   0,
    0,   0,   0,  121, 170, 84,  203, 127, 244, 131, 204, 143, 148, 227, 186, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   88,  81,  165, 25,  115, 208,
    243, 197, 82, 99,  216, 177, 0,   0,   0,   0,   0,   0,   0,   0,   38,  232, 244, 17,  186, 31,  120, 1,   35,
    102, 119, 11, 0,   0,   0,   0,   0,   0,   0,   0,   23,  207, 141, 117, 169, 163, 192, 95,  178, 36,  162, 48,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   77, 72,  9,   148, 254, 79,  175, 192, 243, 29,  147, 147, 0,   0,   0,   0,   0,   0,   0,   0,   79,
    132, 105, 42, 65,  231, 169, 57,  174, 82,  91,  128, 0,   0,   0,   0,   0,   0,   0,   0,   85,  48,  49,  69,
    197, 94,  0,  113, 93,  131, 98,  42,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,   0,   0,   0};
  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NCHW;
  test_main_space_to_batch_nd<float>(input_data, correct_data, input_shape, param, data_type, format);
}
}  // namespace mindspore
