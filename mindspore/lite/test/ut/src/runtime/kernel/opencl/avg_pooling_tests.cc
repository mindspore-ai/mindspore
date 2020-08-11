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
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/pooling2d.h"

namespace mindspore {

class TestAvgPoolingOpenCL : public mindspore::CommonTest {};

void InitAvgPoolingParam(PoolingParameter *param) {
  param->input_batch_ = 1;
  param->input_h_ = 2;
  param->input_w_ = 2;
  param->input_channel_ = 4;

  param->output_batch_ = 1;
  param->output_h_ = 1;
  param->output_w_ = 1;
  param->output_channel_ = 4;

  param->window_h_ = 2;
  param->window_w_ = 2;

  param->stride_h_ = 2;
  param->stride_w_ = 2;

  param->pad_u_ = 0;
  param->pad_d_ = 0;
  param->pad_l_ = 0;
  param->pad_r_ = 0;

  param->max_pooling_ = false;
  param->avg_pooling_ = true;
}

TEST_F(TestAvgPoolingOpenCL, AvgPoolFp32) {
  MS_LOG(INFO) << "start TEST_F TestPoolingOpenCL";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();

  MS_LOG(INFO) << "create PoolingParameter";
  auto param = new PoolingParameter();
  InitAvgPoolingParam(param);

  MS_LOG(INFO) << "create Tensors";
  std::vector<int> shape_in = {
    param->input_batch_,
    param->input_h_,
    param->input_w_,
    param->input_channel_,
  };
  std::vector<int> shape_out = {
    param->output_batch_,
    param->output_h_,
    param->output_w_,
    param->output_channel_,
  };
  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  lite::tensor::Tensor *tensor_in = new lite::tensor::Tensor(data_type, shape_in, schema::Format_NHWC, tensorType);
  lite::tensor::Tensor *tensor_out = new lite::tensor::Tensor(data_type, shape_out, schema::Format_NHWC, tensorType);
  std::vector<lite::tensor::Tensor *> inputs{tensor_in};
  std::vector<lite::tensor::Tensor *> outputs{tensor_out};

  MS_LOG(INFO) << "create OpenCL Kernel";
  auto *pooling_kernel = new kernel::PoolingOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  pooling_kernel->Init();
  std::vector<kernel::LiteKernel *> kernels{pooling_kernel};

  MS_LOG(INFO) << "create SubGraphOpenCLKernel";
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  pGraph->Init();

  MS_LOG(INFO) << "initialize data";
  std::vector<lite::tensor::Tensor *> tensor_map = {tensor_in};
  for (auto &tensor_file : tensor_map) {
    auto tensor = tensor_file;
    size_t size = tensor->Size();
    const float data[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    memcpy(tensor->Data(), data, size);
  }

  MS_LOG(INFO) << "pGraph->Run()";
  pGraph->Run();

  MS_LOG(INFO) << "==================output data=================";
  float *output_data = reinterpret_cast<float *>(tensor_out->Data());
  printf("output:");
  for (int i = 0; i < 4; i++) {
    printf("%.3f ", output_data[i]);
  }
  printf("\n");
  size_t output_size = tensor_out->Size();
  float expect[4] = {2.0f, 3.0f, 4.0f, 5.0f};

  for (int i = 0; i < tensor_out->ElementsNum(); ++i)
    if (std::fabs(output_data[i] - expect[i]) > 1e-5) {
      printf("idx[%d] except=%.3f output=%.3f, ", i, expect[i], output_data[i]);
    }
  printf("test all close OK!\n");
  lite::CompareOutputData(output_data, expect, 4);
}

}  // namespace mindspore
