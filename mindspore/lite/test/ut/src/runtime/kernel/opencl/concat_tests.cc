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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/concat.h"

namespace mindspore {
class TestConcatOpenCLfp32 : public mindspore::CommonTest {
 public:
  TestConcatOpenCLfp32() {}
};
class TestConcatOpenCLfp16 : public mindspore::CommonTest {
 public:
  TestConcatOpenCLfp16() {}
};

class TestConcatOpenCLCI : public mindspore::CommonTest {
 public:
  TestConcatOpenCLCI() {}
};

TEST_F(TestConcatOpenCLCI, ConcatFp32_2inputforCI) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 2;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {std::vector<int>{1, 1, 1, 8}, std::vector<int>{1, 1, 1, 8}};
  std::vector<int> output_shape = {2, 1, 1, 8};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  float input_data1[] = {0.75f, 0.06f, 0.74f, 0.30f, 0.9f, 0.59f, 0.03f, 0.37f};
  float input_data2[] = {0.5f, 0.6f, 0.74f, 0.23f, 0.46f, 0.69f, 0.13f, 0.47f};
  float correctOutput[] = {0.75f, 0.06f, 0.74f, 0.30f, 0.9f,  0.59f, 0.03f, 0.37f,
                           0.5f,  0.6f,  0.74f, 0.23f, 0.46f, 0.69f, 0.13f, 0.47f};
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
  auto param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
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
  param->axis_ = 0;
  auto *concat_kernel =
    new (std::nothrow) kernel::ConcatOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (concat_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ConcatOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  concat_kernel->Init();
  // to do allocate memory for inputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{concat_kernel};
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
    delete concat_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));

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

TEST_F(TestConcatOpenCLfp16, ConcatFp16_4input_dim4_axis1) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, input2_size, input3_size, input4_size, output_size;
  std::string input1Ppath = "./test_data/concatfp16_input1.bin";
  std::string input2Ppath = "./test_data/concatfp16_input2.bin";
  std::string input3Ppath = "./test_data/concatfp16_input3.bin";
  std::string input4Ppath = "./test_data/concatfp16_input4.bin";
  std::string correctOutputPath = "./test_data/concatfp16_output.bin";
  auto input_data1 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto input_data3 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input3Ppath.c_str(), &input3_size));
  auto input_data4 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input4Ppath.c_str(), &input4_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 4;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
    std::vector<int>{1, 19, 19, 96}, std::vector<int>{1, 19, 19, 96}, std::vector<int>{1, 19, 19, 96},
    std::vector<int>{1, 19, 19, 96}};
  std::vector<int> output_shape = {1, 76, 19, 96};
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
  auto param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
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
  param->axis_ = 1;
  auto *concat_kernel =
    new (std::nothrow) kernel::ConcatOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (concat_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ConcatOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  concat_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{concat_kernel};
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
    delete concat_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  if (inputs.size() == 2) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
  } else if (inputs.size() == 3) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
  } else if (inputs.size() == 4) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
    memcpy(inputs[3]->data_c(), input_data4, input4_size);
  } else {
    MS_LOG(ERROR) << " input size must be 2 or 3 or 4";
  }

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
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

TEST_F(TestConcatOpenCLfp32, ConcatFp32_3input_dim4_axis1) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, input2_size, input3_size, output_size;
  std::string input1Ppath = "./test_data/concatfp32_input1.bin";
  std::string input2Ppath = "./test_data/concatfp32_input2.bin";
  std::string input3Ppath = "./test_data/concatfp32_input3.bin";
  std::string correctOutputPath = "./test_data/concatfp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto input_data3 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input3Ppath.c_str(), &input3_size));
  auto correctOutput = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 3;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
    std::vector<int>{1, 16, 256, 80}, std::vector<int>{1, 16, 256, 80}, std::vector<int>{1, 16, 256, 80}};
  std::vector<int> output_shape = {1, 48, 256, 80};
  auto data_type = kNumberTypeFloat32;
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
  MS_LOG(INFO) << " input_shapes size=: " << input_shapes.size();

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
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
  param->axis_ = 1;
  auto *concat_kernel =
    new (std::nothrow) kernel::ConcatOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (concat_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ConcatOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  concat_kernel->Init();
  // to do allocate memory for inputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{concat_kernel};
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
    delete concat_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  if (inputs.size() == 2) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
  } else if (inputs.size() == 3) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
  } else {
    MS_LOG(ERROR) << " input size must be 2 or 3 ";
  }

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

TEST_F(TestConcatOpenCLfp16, ConcatFp16_6input_dim4_axis1) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // get the input from .bin
  size_t input1_size, input2_size, input3_size, input4_size, input5_size, input6_size, output_size;
  std::string input1Ppath = "./test_data/concatfp16_input1.bin";
  std::string input2Ppath = "./test_data/concatfp16_input2.bin";
  std::string input3Ppath = "./test_data/concatfp16_input3.bin";
  std::string input4Ppath = "./test_data/concatfp16_input4.bin";
  std::string input5Ppath = "./test_data/concatfp16_input5.bin";
  std::string input6Ppath = "./test_data/concatfp16_input6.bin";
  std::string correctOutputPath = "./test_data/concatfp16_output.bin";
  auto input_data1 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto input_data3 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input3Ppath.c_str(), &input3_size));
  auto input_data4 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input4Ppath.c_str(), &input4_size));
  auto input_data5 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input5Ppath.c_str(), &input5_size));
  auto input_data6 = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input6Ppath.c_str(), &input6_size));
  auto correctOutput =
    reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  MS_LOG(INFO) << " init tensors ";
  constexpr int INPUT_NUM = 6;
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
    std::vector<int>{1, 1200, 3, 4}, std::vector<int>{1, 600, 3, 4}, std::vector<int>{1, 150, 3, 4},
    std::vector<int>{1, 50, 3, 4},   std::vector<int>{1, 30, 3, 4},  std::vector<int>{1, 4, 3, 4}};
  std::vector<int> output_shape = {1, 2034, 3, 4};
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
  auto param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
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
  param->axis_ = 1;
  auto *concat_kernel =
    new (std::nothrow) kernel::ConcatOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (concat_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::ConcatOpenCLKernel failed ";
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  concat_kernel->Init();
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{concat_kernel};
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
    delete concat_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  if (inputs.size() == 2) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
  } else if (inputs.size() == 3) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
  } else if (inputs.size() == 4) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
    memcpy(inputs[3]->data_c(), input_data4, input4_size);
  } else if (inputs.size() == 6) {
    memcpy(inputs[0]->data_c(), input_data1, input1_size);
    memcpy(inputs[1]->data_c(), input_data2, input2_size);
    memcpy(inputs[2]->data_c(), input_data3, input3_size);
    memcpy(inputs[3]->data_c(), input_data4, input4_size);
    memcpy(inputs[4]->data_c(), input_data5, input5_size);
    memcpy(inputs[5]->data_c(), input_data6, input6_size);
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
