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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/slice.h"

namespace mindspore {
class TestSliceOpenCLfp32 : public mindspore::CommonTest {
 public:
  TestSliceOpenCLfp32() {}
};
class TestSliceOpenCLfp16 : public mindspore::CommonTest {
 public:
  TestSliceOpenCLfp16() {}
};

template <typename T>
void CompareOutputData1(T *output_data, T *correct_data, int size, float err_bound) {
  for (size_t i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}

TEST_F(TestSliceOpenCLfp32, Slicefp32CI) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 2, 2, 8};
  std::vector<int> output_shape = {1, 2, 2, 5};
  std::vector<int> begin = {0, 0, 0, 2};
  std::vector<int> size = {1, 2, 2, 5};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::TensorCategory(schema::NodeType_ValueNode);

  float input_data[] = {-0.45816937, 0.92391545,  -0.9135602, -1.4002057, 1.1080881,  0.40712625,  -0.28128958,
                        0.09470133,  0.19801073,  0.04927751, -1.2808367, 0.1470597,  0.03393711,  -0.33282498,
                        -1.0433807,  -1.3678077,  -0.6423931, 0.5584889,  0.28965706, 0.5343769,   0.75480366,
                        -1.9328151,  -0.48714373, 1.711132,   -1.8871949, -0.2987629, -0.14000037, -0.080552,
                        0.95056856,  -0.06886655, 0.5316237,  0.05787678};
  float correct_data[] = {-0.9135602,  -1.4002057,  1.1080881,  0.40712625, -0.28128958, -1.2808367, 0.1470597,
                          0.03393711,  -0.33282498, -1.0433807, 0.28965706, 0.5343769,   0.75480366, -1.9328151,
                          -0.48714373, -0.14000037, -0.080552,  0.95056856, -0.06886655, 0.5316237};
  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    delete tensor_data;
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data};
  std::vector<lite::Tensor *> outputs = {output_tensor};

  MS_LOG(INFO) << "setting  SliceParameter ";
  auto param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (param == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    MS_LOG(INFO) << "new SliceParameter failed ";
    return;
  }
  for (int i = 0; i < input_shape.size(); i++) {
    param->begin_[i] = begin[i];
    param->size_[i] = size[i];
  }

  auto *slice_kernel =
    new (std::nothrow) kernel::SliceOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (slice_kernel == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    MS_LOG(INFO) << "new kernel::slice_kernel failed ";
    return;
  }
  slice_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{slice_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete slice_kernel;
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, sizeof(input_data));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  CompareOutputData1(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001);
  lite::opencl::OpenCLRuntime::DeleteInstance();
  for (auto tensor : inputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  delete sub_graph;
}

TEST_F(TestSliceOpenCLfp32, Slicefp32input_dim4) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 19, 19, 96};
  std::vector<int> output_shape = {1, 10, 10, 13};
  std::vector<int> begin = {0, 2, 3, 4};
  std::vector<int> size = {1, 10, 10, 13};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::TensorCategory(schema::NodeType_ValueNode);

  // get the input from .bin
  size_t input_size, output_size;
  std::string input_path = "./test_data/in_slicefp32.bin";
  std::string output_path = "./test_data/out_slicefp32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));
  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  if (output_tensor == nullptr) {
    delete tensor_data;
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data};
  std::vector<lite::Tensor *> outputs = {output_tensor};

  MS_LOG(INFO) << "setting  SliceParameter ";
  auto param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (param == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    MS_LOG(INFO) << "new SliceParameter failed ";
    return;
  }
  for (int i = 0; i < input_shape.size(); i++) {
    param->begin_[i] = begin[i];
    param->size_[i] = size[i];
  }

  auto *slice_kernel =
    new (std::nothrow) kernel::SliceOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (slice_kernel == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    MS_LOG(INFO) << "new kernel::slice_kernel failed ";
    return;
  }
  slice_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{slice_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete slice_kernel;
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->data_c());
  CompareOutputData1(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001);
  lite::opencl::OpenCLRuntime::DeleteInstance();
  for (auto tensor : inputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  delete sub_graph;
}

TEST_F(TestSliceOpenCLfp16, Slicefp16input_dim4) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->SetFp16Enable(true);
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << " Read tensors from .bin ";
  std::vector<int> input_shape = {1, 25, 25, 48};
  std::vector<int> output_shape = {1, 24, 24, 15};
  std::vector<int> begin = {0, 1, 1, 7};
  std::vector<int> size = {1, 24, 24, 15};
  auto data_type = kNumberTypeFloat16;
  auto tensor_type = lite::TensorCategory(schema::NodeType_ValueNode);

  // get the input from .bin
  size_t input_size, output_size;
  std::string input_path = "./test_data/in_slicefp16.bin";
  std::string output_path = "./test_data/out_slicefp16.bin";
  auto input_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float16_t *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));

  MS_LOG(INFO) << " construct tensors ";
  lite::Tensor *tensor_data = new (std::nothrow) lite::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr) {
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  auto *output_tensor = new (std::nothrow) lite::Tensor(data_type, output_shape, schema::Format_NHWC4, tensor_type);
  if (output_tensor == nullptr) {
    delete tensor_data;
    MS_LOG(INFO) << " init tensor failed ";
    return;
  }
  std::vector<lite::Tensor *> inputs = {tensor_data};
  std::vector<lite::Tensor *> outputs = {output_tensor};

  MS_LOG(INFO) << " setting  SliceParameter ";
  auto param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (param == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    MS_LOG(INFO) << " new SliceParameter failed ";
    return;
  }
  for (int i = 0; i < input_shape.size(); i++) {
    param->begin_[i] = begin[i];
    param->size_[i] = size[i];
  }

  auto *slice_kernel =
    new (std::nothrow) kernel::SliceOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (slice_kernel == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    MS_LOG(INFO) << " new kernel::slice_kernel failed ";
    return;
  }
  slice_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{slice_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    delete param;
    delete slice_kernel;
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  memcpy(inputs[0]->data_c(), input_data, input_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float16_t *>(output_tensor->data_c());
  CompareOutputData1(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001);
  lite::opencl::OpenCLRuntime::DeleteInstance();
  for (auto tensor : inputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    tensor->SetData(nullptr);
    delete tensor;
  }
  delete sub_graph;
}
}  // namespace mindspore
