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
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/hswish.h"
using mindspore::lite::Tensor;
using mindspore::schema::Format::Format_NHWC;
namespace mindspore {
class TestSwishOpenCLCI : public mindspore::CommonTest {
 public:
  TestSwishOpenCLCI() {}
};

TEST_F(TestSwishOpenCLCI, Fp32CI) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape = {2, 10, 1, 4};
  std::vector<int> output_shape = {2, 10, 1, 4};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  float input_data[] = {2.5f,  6.0f,  -7.4f, -3.5f, 5.9f,  6.5f,  -8.0f, 7.4f,  5.9f,  6.5f,  -8.0f, 7.4f,  7.5f,  6.0f,
                        -7.4f, -3.5f, 7.5f,  6.0f,  -7.4f, -3.5f, 5.9f,  6.5f,  -8.0f, 7.4f,  5.9f,  6.5f,  -8.0f, 7.4f,
                        7.5f,  6.0f,  -7.4f, -3.5f, 7.5f,  6.0f,  -7.4f, -3.5f, 5.9f,  6.5f,  -8.0f, 7.4f,  5.9f,  6.5f,
                        -8.0f, 7.4f,  7.5f,  6.0f,  -7.4f, -3.5f, 7.5f,  6.0f,  -7.4f, -3.5f, 5.9f,  6.5f,  -8.0f, 7.4f,
                        5.9f,  6.5f,  -8.0f, 7.4f,  7.5f,  6.0f,  -7.4f, -3.5f, 7.5f,  6.0f,  -7.4f, -3.5f, 5.9f,  6.5f,
                        -8.0f, 7.4f,  5.9f,  6.5f,  -8.0f, 7.4f,  7.5f,  6.0f,  -7.4f, -3.5f};

  float correctOutput[] = {0.9167f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                           0.0f,    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
                           1.0f,    1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                           0.0f,    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
                           1.0f,    1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                           0.0f,    1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
  auto output_tensor = Tensor(data_type, input_shape, Format_NHWC, tensor_type);
  auto in_tensor = Tensor(data_type, output_shape, Format_NHWC, tensor_type);
  std::vector<lite::Tensor *> inputs{&in_tensor};
  std::vector<lite::Tensor *> outputs{&output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *hswish_kernel =
    new (std::nothrow) kernel::HswishOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (hswish_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::HswishOpenCLKernel failed ";
    delete param;
    return;
  }
  hswish_kernel->Init();
  // to do allocate memory for inputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{hswish_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete hswish_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data, sizeof(input_data));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}
}  // namespace mindspore
