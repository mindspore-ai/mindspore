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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/batchnorm.h"

namespace mindspore {
class TestBatchnormOpenCLfp32 : public mindspore::CommonTest {
 public:
  TestBatchnormOpenCLfp32() {}
};
class TestBatchnormOpenCLfp16 : public mindspore::CommonTest {
 public:
  TestBatchnormOpenCLfp16() {}
};
class TestBatchnormOpenCLCI : public mindspore::CommonTest {
 public:
  TestBatchnormOpenCLCI() {}
};

TEST_F(TestBatchnormOpenCLCI, Batchnormfp32CI) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 2, 2, 8};
  std::vector<int> output_shape = {1, 2, 2, 8};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;

  float input_data[] = {2.471454,   -2.1379554,  -0.0904604, 1.2928944,  -0.19215967, -0.8677279, -0.12759617,
                        1.2242758,  -0.06398406, -0.4041858, 0.20352598, -2.067808,   0.52113044, -1.567617,
                        0.28003863, 0.41367245,  0.77298605, 0.29908583, 1.4015813,   1.330567,   1.760135,
                        0.6320845,  0.6995399,   -1.208123,  -1.9738104, -1.3283046,  1.022744,   0.02741058,
                        0.84505165, -0.89434445, 1.983211,   -0.5485428};
  float correct_data[] = {0.7505676,  0.515882,   0.26147857, 1.6026789,  0.47575232, 0.50116986, 0.33589783,
                          1.4884706,  0.56019205, 0.7832671,  0.53893626, -0.5093127, 0.71395767, 0.18509413,
                          0.33990562, 0.891792,   0.6230367,  0.89172685, 1.6696336,  1.6263539,  1.1277269,
                          1.1784974,  0.34403008, -0.3019984, 0.4167911,  0.6407478,  1.3120956,  0.80740136,
                          0.8221321,  0.4891496,  0.3566509,  0.18351318};
  float mean_data[] = {0.3016613, -0.89284, 0.63434774, 0.145766, 0.73353934, -0.6744012, 0.7087985, -0.02967937};
  float var_data[] = {2.5604038, 0.84985304, 0.36261332, 1.9083935, 0.4920925, 0.6476224, 0.6269014, 0.8567283};
  float scale_data[] = {0.1201471, 0.142174, 0.5683258, 0.86815494, 0.23426804, 0.3634345, 0.0077846, 0.6813278};
  float offset_data[] = {0.58764684, 0.70790595, 0.945536, 0.8817803, 0.78489226, 0.5884778, 0.3441211, 0.5654443};

  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_mean =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_var =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_scale =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_offset =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr || tensor_mean == nullptr || tensor_var == nullptr || tensor_scale == nullptr ||
      tensor_offset == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    delete tensor_data;
    delete tensor_mean;
    delete tensor_var;
    delete tensor_scale;
    delete tensor_offset;
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data, tensor_scale, tensor_offset, tensor_mean, tensor_var};
  std::vector<lite::Tensor *> outputs{output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new BatchNormParameter failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->epsilon_ = pow(10, -5);
  auto *batchnorm_kernel =
    new (std::nothrow) kernel::BatchNormOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (batchnorm_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::BatchNorm_kernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  batchnorm_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{batchnorm_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete batchnorm_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, sizeof(input_data));
  memcpy(inputs[1]->data_c(), scale_data, sizeof(scale_data));
  memcpy(inputs[2]->data_c(), offset_data, sizeof(offset_data));
  memcpy(inputs[3]->data_c(), mean_data, sizeof(mean_data));
  memcpy(inputs[4]->data_c(), var_data, sizeof(var_data));
  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001));
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

TEST_F(TestBatchnormOpenCLfp16, Batchnormfp16input_dim4) {
  MS_LOG(INFO) << "begin test";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 256, 256, 48};
  std::vector<int> output_shape = {1, 256, 256, 48};
  auto data_type = kNumberTypeFloat16;
  auto tensor_type = lite::Tensor::CONST_TENSOR;

  // get the input from .bin
  size_t input_size, output_size;
  std::string input_path = "./test_data/batchnorm_in_datafp16.bin";
  std::string mean_path = "./test_data/batchnorm_meanfp16.bin";
  std::string var_path = "./test_data/batchnorm_varfp16.bin";
  std::string offset_path = "./test_data/batchnorm_offsetfp16.bin";
  std::string scale_path = "./test_data/batchnorm_scalefp16.bin";
  std::string output_path = "./test_data/batchnorm_correctdatafp16.bin";
  auto input_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  size_t mean_size, var_size, scale_size, offset_size;
  auto mean_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(mean_path.c_str(), &mean_size));
  auto var_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(var_path.c_str(), &var_size));
  auto scale_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(scale_path.c_str(), &scale_size));
  auto offset_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(offset_path.c_str(), &offset_size));

  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_mean =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_var =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_scale =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_offset =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr || tensor_mean == nullptr || tensor_var == nullptr || tensor_scale == nullptr ||
      tensor_offset == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC4, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    delete tensor_data;
    delete tensor_mean;
    delete tensor_var;
    delete tensor_scale;
    delete tensor_offset;
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data, tensor_scale, tensor_offset, tensor_mean, tensor_var};
  std::vector<lite::Tensor *> outputs{output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new BatchNormParameter failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->epsilon_ = pow(10, -5);
  auto *batchnorm_kernel =
    new (std::nothrow) kernel::BatchNormOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (batchnorm_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::BatchNorm_kernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  batchnorm_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{batchnorm_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete batchnorm_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);
  memcpy(inputs[1]->data_c(), scale_data, scale_size);
  memcpy(inputs[2]->data_c(), offset_data, offset_size);
  memcpy(inputs[3]->data_c(), mean_data, mean_size);
  memcpy(inputs[4]->data_c(), var_data, var_size);
  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.01));
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

TEST_F(TestBatchnormOpenCLfp32, Batchnormfp32input_dim4) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 256, 256, 47};
  std::vector<int> output_shape = {1, 256, 256, 47};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;

  // get the input from .bin
  size_t input_size, output_size;
  std::string input_path = "./test_data/batchnorm_in_datafp32.bin";
  std::string mean_path = "./test_data/batchnorm_meanfp32.bin";
  std::string var_path = "./test_data/batchnorm_varfp32.bin";
  std::string offset_path = "./test_data/batchnorm_offsetfp32.bin";
  std::string scale_path = "./test_data/batchnorm_scalefp32.bin";
  std::string output_path = "./test_data/batchnorm_out_datafp32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  size_t mean_size, var_size, scale_size, offset_size;
  auto mean_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(mean_path.c_str(), &mean_size));
  auto var_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(var_path.c_str(), &var_size));
  auto scale_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(scale_path.c_str(), &scale_size));
  auto offset_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(offset_path.c_str(), &offset_size));

  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_mean =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_var =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_scale =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  lite::Tensor *tensor_offset =
    new (std::nothrow) lite::Tensor(data_type, {1, 1, 1, input_shape[3]}, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr || tensor_mean == nullptr || tensor_var == nullptr || tensor_scale == nullptr ||
      tensor_offset == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    delete tensor_data;
    delete tensor_mean;
    delete tensor_var;
    delete tensor_scale;
    delete tensor_offset;
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data, tensor_scale, tensor_offset, tensor_mean, tensor_var};
  std::vector<lite::Tensor *> outputs{output_tensor};

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new BatchNormParameter failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    return;
  }
  param->epsilon_ = pow(10, -5);
  auto *batchnorm_kernel =
    new (std::nothrow) kernel::BatchNormOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (batchnorm_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::BatchNorm_kernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    return;
  }
  batchnorm_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{batchnorm_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete batchnorm_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);
  memcpy(inputs[1]->data_c(), scale_data, scale_size);
  memcpy(inputs[2]->data_c(), offset_data, offset_size);
  memcpy(inputs[3]->data_c(), mean_data, mean_size);
  memcpy(inputs[4]->data_c(), var_data, var_size);
  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001));
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
