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
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/arithmetic_self.h"

namespace mindspore {
class TestArithmeticSelfOpenCLfp16 : public mindspore::CommonTest {
 public:
  TestArithmeticSelfOpenCLfp16() {}
};

class TestArithmeticSelfOpenCLCI : public mindspore::CommonTest {
 public:
  TestArithmeticSelfOpenCLCI() {}
};

template <typename T>
void CompareOutputData1(T *input_data1, T *output_data, T *correct_data, int size, float err_bound) {
  for (size_t i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}

TEST_F(TestArithmeticSelfOpenCLfp16, ArithmeticSelfOpenCLFp16) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, output_size;
  std::string input1Ppath = "./test_data/in_arithmetic_selffp16.bin";
  std::string correctOutputPath = "./test_data/out_arithmetic_selffp16.bin";
  auto input_data1 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";

  std::vector<int> shape = {1, 2, 2, 144};
  auto data_type = kNumberTypeFloat16;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  auto *input_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    MS_LOG(INFO) << " new input_tensor or output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs{input_tensor};
  std::vector<lite::Tensor *> outputs{output_tensor};

  MS_LOG(INFO) << " initialize param ";
  auto param = reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ConcatParameter failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Sin;
  auto *arithmeticself_kernel =
    new (std::nothrow) kernel::ArithmeticSelfOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (arithmeticself_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ArithmeticSelfOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  arithmeticself_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{arithmeticself_kernel};
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
    delete arithmeticself_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, input1_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
  CompareOutputData1(input_data1, output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001);
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

TEST_F(TestArithmeticSelfOpenCLCI, ArithmeticSelfRound) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();
  float input_data1[] = {0.75f, 0.06f, 0.74f, 0.30f, 0.9f, 0.59f, 0.03f, 0.37f,
                         0.75f, 0.06f, 0.74f, 0.30f, 0.9f, 0.59f, 0.03f, 0.37f};
  float correctOutput[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape = {1, 1, 4, 4};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  auto *input_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NHWC, tensor_type);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    MS_LOG(INFO) << " new input_tensor or output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs{input_tensor};
  std::vector<lite::Tensor *> outputs{output_tensor};

  MS_LOG(INFO) << " initialize param ";
  auto param = reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ConcatParameter failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Round;
  auto *arithmeticself_kernel =
    new (std::nothrow) kernel::ArithmeticSelfOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (arithmeticself_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ArithmeticSelfOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  arithmeticself_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{arithmeticself_kernel};
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
    delete arithmeticself_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  std::cout << sizeof(input_data1) / sizeof(float) << std::endl;
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  CompareOutputData1(input_data1, output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001);
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

TEST_F(TestArithmeticSelfOpenCLfp16, ArithmeticSelfdim2Fp16) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, output_size;
  std::string input1Ppath = "./test_data/in_arithmetic_selffp16.bin";
  std::string correctOutputPath = "./test_data/out_arithmetic_selffp16.bin";
  auto input_data1 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";

  std::vector<int> shape = {1, 512};
  auto data_type = kNumberTypeFloat16;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  auto *input_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NC, tensor_type);
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, shape, schema::Format_NC, tensor_type);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    MS_LOG(INFO) << " new input_tensor or output_tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs{input_tensor};
  std::vector<lite::Tensor *> outputs{output_tensor};
  MS_LOG(INFO) << " initialize param ";
  auto param = reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ConcatParameter failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Sin;
  auto *arithmeticself_kernel =
    new (std::nothrow) kernel::ArithmeticSelfOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (arithmeticself_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ArithmeticSelfOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  arithmeticself_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{arithmeticself_kernel};
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
    delete arithmeticself_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, input1_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
  CompareOutputData1(input_data1, output_data_gpu, correctOutput, output_tensor->ElementsNum(), 0.000001);
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
