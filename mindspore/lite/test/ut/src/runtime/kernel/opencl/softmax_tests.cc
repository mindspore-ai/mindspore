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
  std::cout << "runtime" << std::endl;
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // define tensor
  MS_LOG(INFO) << "defineTensor";
  std::cout << "defineTensor" << std::endl;

  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  auto input_tensor = new lite::tensor::Tensor(data_type, input_shape, format, tensorType);
  auto output_tensor = new lite::tensor::Tensor(data_type, output_shape, format, tensorType);
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  // run
  MS_LOG(INFO) << "NewOpenCLKernel";
  std::cout << "NewOpenCLKernel" << std::endl;
  auto *kernel = new kernel::SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  MS_LOG(INFO) << "KernelInit";
  std::cout << "KernelInit" << std::endl;
  kernel->Init();

  std::cout << "LiteKernel" << std::endl;
  std::vector<kernel::LiteKernel *> kernels{kernel};
  inputs[0]->MallocData(allocator);
  std::cout << "SubGraphOpenCLKernel" << std::endl;
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
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
  std::cout << "compare result" << std::endl;
  CompareOutput(output_tensor, expect_file);
}

TEST_F(TestSoftmaxOpenCL, Softmax_1) {
  std::vector<int> input_shape = {1, 2, 2, 8};
  std::vector<int> output_shape = {1, 2, 2, 8};
  std::string input_file = "softmax_in.bin";
  std::string expect_file = "softmax_out.bin";
  auto param = new SoftmaxParameter;
  param->axis_ = 3;
  schema::Format format = schema::Format_NHWC4;

  RunTestCase(input_shape, output_shape, input_file, expect_file, param, format);
}

// TEST_F(TestSoftmaxOpenCL, Softmax_1x1) {
//  std::vector<int> input_shape = {1, 100};
//  std::vector<int> output_shape = {1, 100};
//  std::string input_file = "softmax1x1_in.bin";
//  std::string expect_file = "softmax1x1_out.bin";
//  auto param = new SoftmaxParameter;
//  param->axis_ = 1;
//  schema::Format format = schema::Format_NHWC4;
//
//  RunTestCase(input_shape, output_shape, input_file, expect_file, param, format);
//}

}  // namespace mindspore
