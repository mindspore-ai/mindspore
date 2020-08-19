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
#include "mindspore/core/utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/softmax.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {

class TestSoftmaxOpenCL : public mindspore::CommonTest {};

void RunTestCase(std::vector<int> input_shape, std::vector<int> output_shape, std::string input_file,
                 std::string expect_file, SoftmaxParameter *param, schema::Format format) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // define tensor
  MS_LOG(INFO) << "defineTensor";
  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  auto input_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, input_shape, format, tensorType);
  auto output_tensor = new (std::nothrow) lite::tensor::Tensor(data_type, output_shape, format, tensorType);
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor null";
    return;
  }
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor null";
    return;
  }
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  // run
  MS_LOG(INFO) << "NewOpenCLKernel";
  auto *kernel = new kernel::SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel null";
    return;
  }
  MS_LOG(INFO) << "KernelInit";
  kernel->Init();

  std::vector<kernel::LiteKernel *> kernels{kernel};
  inputs[0]->MallocData(allocator);
  auto *pGraph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph null";
    return;
  }
  MS_LOG(INFO) << "pGraphinit";
  pGraph->Init();

  // load data
  MS_LOG(INFO) << "load data1";
  LoadTestData(input_tensor->Data(), input_tensor->Size(), input_file);
  auto *input_data = reinterpret_cast<float *>(input_tensor->Data());
  printf("\ninput[0:10]:");
  for (int i = 0; i < 10; i++) {
    printf("[%d]:%.3f ", i, input_data[i]);
  }
  printf("\n\n");

  MS_LOG(INFO) << "Run";
  pGraph->Run();

  MS_LOG(INFO) << "compare result";
  CompareOutput(output_tensor, expect_file);
}

TEST_F(TestSoftmaxOpenCL, Softmax_1) {
  std::vector<int> input_shape = {1, 2, 2, 8};
  std::vector<int> output_shape = {1, 2, 2, 8};
  std::string input_file = "softmax_in.bin";
  std::string expect_file = "softmax_out.bin";
  auto param = new (std::nothrow) SoftmaxParameter;
  param->axis_ = 3;
  schema::Format format = schema::Format_NHWC4;

  RunTestCase(input_shape, output_shape, input_file, expect_file, param, format);
}

}  // namespace mindspore
