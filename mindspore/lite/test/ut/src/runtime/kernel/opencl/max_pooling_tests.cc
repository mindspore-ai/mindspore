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

#include <memory>
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/pooling2d.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {

class TestMaxPoolingOpenCL : public mindspore::CommonTest {};

void InitParameter(PoolingParameter *param) {
  param->window_h_ = 2;
  param->window_w_ = 2;
  param->stride_h_ = 2;
  param->stride_w_ = 2;
  param->pad_u_ = 0;
  param->pad_d_ = 0;
  param->pad_l_ = 0;
  param->pad_r_ = 0;
  param->avg_pooling_ = false;
  param->max_pooling_ = true;
}

TEST_F(TestMaxPoolingOpenCL, MaxPool_1_32_512_96) {
  MS_LOG(INFO) << "ocl runtime";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "PoolingParameter";
  auto param = new PoolingParameter;
  InitParameter(param);

  // define tensor
  MS_LOG(INFO) << "define tensor1";
  std::vector<int> input_shape = {1, 16, 256, 192};
  std::vector<int> output_shape = {1, 8, 128, 192};
  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  MS_LOG(INFO) << "define tensor2";
  auto input_tensor = new lite::tensor::Tensor(data_type, input_shape, schema::Format_NHWC4, tensorType);
  auto output_tensor = new lite::tensor::Tensor(data_type, output_shape, schema::Format_NHWC4, tensorType);
  MS_LOG(INFO) << "define input";
  std::vector<lite::tensor::Tensor *> inputs{input_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  // run
  MS_LOG(INFO) << "pooling_kernel";
  auto *pooling_kernel = new kernel::PoolingOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  MS_LOG(INFO) << "pooling_kernel init";
  pooling_kernel->Init();

  std::vector<kernel::LiteKernel *> kernels{pooling_kernel};
  inputs[0]->MallocData(allocator);
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  MS_LOG(INFO) << "pGraph init";
  pGraph->Init();

  // load data
  MS_LOG(INFO) << "load data1";
  std::string input_file = "maxpool_in.bin";
  std::string expect_file = "maxpool_out.bin";
  MS_LOG(INFO) << "load data2";
  LoadTestData(input_tensor->Data(), input_tensor->Size(), input_file);
  auto *input_data = reinterpret_cast<float *>(input_tensor->Data());
  printf("input[0:10]:");
  for (int i = 0; i < 10; i++) {
    printf("[%d]:%.3f ", i, input_data[i]);
  }
  printf("\n");

  pGraph->Run();

  MS_LOG(INFO) << "compare result";
  std::cout << "compare result" << std::endl;
  CompareOutput(output_tensor, expect_file);
}

}  // namespace mindspore
