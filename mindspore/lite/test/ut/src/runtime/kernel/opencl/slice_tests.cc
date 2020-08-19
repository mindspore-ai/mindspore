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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/slice.h"

namespace mindspore {
class TestSliceOpenCL : public mindspore::CommonTest {
 public:
  TestSliceOpenCL() {}
};

template <typename T>
void CompareOutputData1(T *output_data, T *correct_data, int size, float err_bound) {
  for (size_t i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}

TEST_F(TestSliceOpenCL, Sliceinput_dim4) {
  MS_LOG(INFO) << "begin test";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "Read tensors from .bin";
  std::vector<int> input_shape = {1, 256, 256, 48};
  std::vector<int> output_shape = {1, 255, 255, 15};
  std::vector<int> begin = {0, 1, 1, 7};
  std::vector<int> size = {1, 255, 255, 15};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;

  // get the input from .bin
  size_t input_size, output_size;
  std::string input_path = "./test_data/in_data.bin";
  std::string output_path = "./test_data/out_data.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto correct_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(output_path.c_str(), &output_size));

  MS_LOG(INFO) << "construct tensors";
  lite::tensor::Tensor *tensor_data =
    new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC, tensor_type);
  if (tensor_data == nullptr) {
    MS_LOG(INFO) << "init tensor failed";
    return;
  }
  auto *output_tensor =
    new (std::nothrow) lite::tensor::Tensor(data_type, output_shape, schema::Format_NHWC4, tensor_type);
  if (output_tensor == nullptr) {
    delete tensor_data;
    MS_LOG(INFO) << "init tensor failed";
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs = {tensor_data};
  std::vector<lite::tensor::Tensor *> outputs = {output_tensor};

  MS_LOG(INFO) << "setting  SliceParameter";
  auto param = new (std::nothrow) SliceParameter();
  if (param == nullptr) {
    for (auto tensor : inputs) {
      delete tensor;
    }
    for (auto tensor : outputs) {
      delete tensor;
    }
    MS_LOG(INFO) << "new SliceParameter failed";
    return;
  }
  for (int i = 0; i < 4; i++) {
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
    MS_LOG(INFO) << "new kernel::slice_kernel failed";
    return;
  }
  slice_kernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << "initialize sub_graph";
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
    MS_LOG(INFO) << "new kernel::SubGraphOpenCLKernel failed";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << "init tensors";
  memcpy(inputs[0]->Data(), input_data, input_size);

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();

  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->Data());
  CompareOutputData1(output_data_gpu, correct_data, output_tensor->ElementsNum(), 0.0001);
  for (auto tensor : inputs) {
    delete tensor;
  }
  for (auto tensor : outputs) {
    delete tensor;
  }
  delete slice_kernel;
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}
}  // namespace mindspore
