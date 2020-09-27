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
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  std::vector<int> indices_shape = {static_cast<int>(indices.size())};
  std::vector<int> output_shape = input_shape;
  output_shape[param->axis_] = indices.size();

  auto tensor_a = lite::Tensor(TypeId(data_type), input_shape, format);
  auto tensor_b = lite::Tensor(TypeId(data_type), indices_shape, schema::Format_NC);
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
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete pkernel;
    MS_LOG(INFO) << " new SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);

  sub_graph->Run();

  std::cout << "==================output data================" << std::endl;
  auto *output_data = reinterpret_cast<T *>(outputs[0]->data_c());
  CommonTest::CompareOutputData<T>(output_data, static_cast<T *>(correct_data), outputs[0]->ElementsNum(), 0.0001);
  delete sub_graph;
}
TEST_F(TestGatherOpenCL, Axis1Fp32) {
  std::vector<int> input_shape{1, 5, 4, 4};
  std::vector<int> indices{1, 3};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
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
TEST_F(TestGatherOpenCL, Axis2Int32) {
  std::vector<int> input_shape{1, 5, 4, 4};
  std::vector<int> indices{1, 3};
  GatherParameter *param = std::make_unique<GatherParameter>().release();
  param->axis_ = 1;
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
  test_main_gather<int>(input_data, correct_data, input_shape, indices, param, data_type, format);
}
}  // namespace mindspore
