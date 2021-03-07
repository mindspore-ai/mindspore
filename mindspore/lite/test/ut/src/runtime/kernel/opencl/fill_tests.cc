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
#include "mindspore/lite/src/runtime/kernel/opencl/opencl_subgraph.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/fill.h"
using mindspore::lite::Tensor;
using mindspore::schema::PrimitiveType_Fill;
using mindspore::schema::PrimitiveType_Shape;
using mindspore::schema::Format::Format_NHWC;

// PrimitiveType_Fill: src/ops/populate/fill_populate.cc

namespace mindspore::lite::opencl::test {
class TestFillOpenCLCI : public mindspore::CommonTest {
 public:
  TestFillOpenCLCI() {}
};

TEST_F(TestFillOpenCLCI, Fp32testfill) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {2};
  float input_data1[] = {3, 3};
  float correctOutput[] = {9, 9, 9, 9, 9, 9, 9, 9, 9};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {3, 3};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  std::vector<lite::Tensor *> inputs{&in_tensor1};
  std::vector<lite::Tensor *> outputs{&output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<FillParameter *>(malloc(sizeof(FillParameter)));
  param->num_dims_ = 9;
  param->op_parameter_.type_ = PrimitiveType_Fill;
  if (param == nullptr) {
    MS_LOG(INFO) << " new FillParameter failed ";
    return;
  }

  auto *fill_kernel =
    new (std::nothrow) kernel::FillOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, nullptr);
  if (fill_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::FillOpenCLKernel failed ";
    delete param;
    return;
  }
  fill_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{fill_kernel};
  auto *sub_graph = new (std::nothrow) kernel::OpenCLSubGraph({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::OpenCLSubGraph failed ";
    delete param;
    delete fill_kernel;
    return;
  }
  // to allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestFillOpenCLCI, Fp32testshape) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {2, 4};
  float input_data1[] = {-0.4045, -0.0924, -0.617, -0.10114, -0.9893, 0.3342, 2.445, -2.182};
  float correctOutput[] = {2, 4};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {2};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  std::vector<lite::Tensor *> inputs{&in_tensor1};
  std::vector<lite::Tensor *> outputs{&output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  param->type_ = PrimitiveType_Shape;
  if (param == nullptr) {
    MS_LOG(INFO) << " new FillParameter failed ";
    return;
  }

  auto *fill_kernel =
    new (std::nothrow) kernel::FillOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, nullptr);
  if (fill_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::FillOpenCLKernel failed ";
    delete param;
    return;
  }
  fill_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{fill_kernel};
  auto *sub_graph = new (std::nothrow) kernel::OpenCLSubGraph({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::OpenCLSubGraph failed ";
    delete param;
    delete fill_kernel;
    return;
  }
  // to allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}
}  // namespace mindspore::lite::opencl::test
