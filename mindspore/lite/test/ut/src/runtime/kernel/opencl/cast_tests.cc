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
#include "mindspore/lite/src/runtime/kernel/opencl/opencl_subgraph.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/cast.h"

// PrimitiveType_Cast: src/ops/populate/cast_populate.cc

namespace mindspore::lite::opencl::test {
class TestCastSelfOpenCL : public CommonTest {
 public:
  TestCastSelfOpenCL() {}
};

template <typename T>
void CompareOutputData1(T *output_data, T *correct_data, int size, float err_bound) {
  for (size_t i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}

TEST_F(TestCastSelfOpenCL, Castfp32tofp16) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, output_size;
  std::string input1Ppath = "./test_data/in_castfp32.bin";
  std::string correctOutputPath = "./test_data/out_castfp16.bin";

  MS_LOG(INFO) << " initialize param ";
  auto param = reinterpret_cast<CastParameter *>(malloc(sizeof(CastParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new CastParameter failed ";
    return;
  }
  param->src_type_ = kNumberTypeFloat32;
  param->dst_type_ = kNumberTypeFloat16;
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape = {1, 23, 39, 47};
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  auto *input_tensor = new (std::nothrow) lite::Tensor(kNumberTypeFloat32, shape, schema::Format_NHWC, tensor_type);
  auto *output_tensor = new (std::nothrow) lite::Tensor(kNumberTypeFloat16, shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    MS_LOG(INFO) << " new input_tensor or output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs{input_tensor};
  std::vector<lite::Tensor *> outputs{output_tensor};

  auto *cast_kernel =
    new (std::nothrow) kernel::CastOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, nullptr);
  if (cast_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::CastOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  cast_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{cast_kernel};
  auto *sub_graph = new (std::nothrow) kernel::OpenCLSubGraph(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::OpenCLSubGraph failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete cast_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data, input1_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
  CompareOutputData1(output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001);
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

TEST_F(TestCastSelfOpenCL, Castfp16tofp32) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, output_size;
  std::string input1Ppath = "./test_data/in_castfp16.bin";
  std::string correctOutputPath = "./test_data/out_castfp32.bin";

  MS_LOG(INFO) << " initialize param ";
  auto param = reinterpret_cast<CastParameter *>(malloc(sizeof(CastParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new CastParameter failed ";
    return;
  }
  param->src_type_ = kNumberTypeFloat16;
  param->dst_type_ = kNumberTypeFloat32;
  auto input_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto correctOutput = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape = {1, 23, 39, 47};
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  auto *input_tensor = new (std::nothrow) lite::Tensor(kNumberTypeFloat16, shape, schema::Format_NHWC, tensor_type);
  auto *output_tensor = new (std::nothrow) lite::Tensor(kNumberTypeFloat32, shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    MS_LOG(INFO) << " new input_tensor or output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs{input_tensor};
  std::vector<lite::Tensor *> outputs{output_tensor};

  auto *cast_kernel =
    new (std::nothrow) kernel::CastOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, nullptr);
  if (cast_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::CastOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  cast_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{cast_kernel};
  auto *sub_graph = new (std::nothrow) kernel::OpenCLSubGraph(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::OpenCLSubGraph failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete cast_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data, input1_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  CompareOutputData1(output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001);
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
}  // namespace mindspore::lite::opencl::test
