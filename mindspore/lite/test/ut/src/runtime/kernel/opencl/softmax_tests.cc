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
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/softmax.h"

namespace mindspore {

class TestSoftmaxOpenCL : public mindspore::Common {};

void InitSoftaxParam(SoftmaxParameter *param) { param->axis_ = -1; }

TEST_F(TestSoftmaxOpenCL, SoftmaxFp32) {
  std::cout << "======" << std::endl;
  MS_LOG(INFO) << "start TEST_F TestSoftmaxOpenCL";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();

  MS_LOG(INFO) << "create SoftmaxParameter";
  auto param = new SoftmaxParameter();
  InitSoftaxParam(param);

  MS_LOG(INFO) << "create Tensors";
  std::vector<int> shape_in = {1, 2, 2, 1};
  std::vector<int> shape_out = {1, 2, 2, 1};
  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  lite::tensor::Tensor *tensor_in = new lite::tensor::Tensor(data_type, shape_in, schema::Format_NCHW, tensorType);
  lite::tensor::Tensor *tensor_out = new lite::tensor::Tensor(data_type, shape_out, schema::Format_NCHW, tensorType);
  std::vector<lite::tensor::Tensor *> inputs{tensor_in};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};

  MS_LOG(INFO) << "create OpenCL Kernel";
  auto *Softmax_kernel = new kernel::SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  Softmax_kernel->Init();
  std::vector<kernel::LiteKernel *> kernels{Softmax_kernel};

  MS_LOG(INFO) << "create SubGraphOpenCLKernel";
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  pGraph->Init();

  MS_LOG(INFO) << "initialize data";
  std::vector<lite::tensor::Tensor *> tensor_map = {tensor_in};
  for (auto &tensor_file : tensor_map) {
    auto tensor = tensor_file;
    size_t size = tensor->Size();
    const float data[4] = {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f)};
    memcpy(tensor->Data(), data, size);
  }

  MS_LOG(INFO) << "pGraph->Run()";
  pGraph->Run();

  MS_LOG(INFO) << "==================output data=================";
  float *output_data = reinterpret_cast<float *>(tensor_out->Data());
  size_t output_size = tensor_out->Size();

  printf("output:");
  for (int i = 0; i < 4; i++) {
    printf("%.3f ", output_data[i]);
  }
  printf("\n");
  float expect[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  for (int i = 0; i < tensor_out->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect[i]) > 1e-5) {
      printf("idx[%d] except=%.3f output=%.3f .", i, expect[i], output_data[i]);
    }
  }
  printf("\nTest all close OK for %zu!\n", output_size);
  lite::CompareOutputData(output_data, expect, 4);
}

}  // namespace mindspore
