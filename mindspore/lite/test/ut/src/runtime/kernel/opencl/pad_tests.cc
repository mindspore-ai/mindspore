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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/pad.h"
#include "nnacl/pack.h"

using mindspore::kernel::LiteKernel;
using mindspore::kernel::PadOpenCLKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::Tensor;
using mindspore::schema::Format;
using mindspore::schema::Format_NC4HW4;
using mindspore::schema::Format_NHWC;
using mindspore::schema::Format_NHWC4;
using mindspore::schema::NodeType_ValueNode;
using mindspore::schema::PaddingMode;
using mindspore::schema::PaddingMode_CONSTANT;
using mindspore::schema::PaddingMode_REFLECT;
using mindspore::schema::PaddingMode_SYMMETRIC;

namespace mindspore {

class TestPadOpenCL : public mindspore::CommonTest {};

void TEST_MAIN(PadParameter *param, Format input_format, Format output_format, Format op_format, const TypeId data_type,
               const std::vector<int> &input_shape, const std::vector<int> &output_shape, const float *input_data,
               const float *expect_data) {
  auto ocl_runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = ocl_runtime_wrapper.GetInstance();
  ocl_runtime->Init();
  ocl_runtime->SetFp16Enable(data_type == kNumberTypeFloat16);
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(DEBUG) << "create Tensors";
  auto input = Tensor(kNumberTypeFloat32, input_shape, input_format, lite::Tensor::CONST_TENSOR);
  auto output = Tensor(kNumberTypeFloat32, output_shape, output_format, lite::Tensor::CONST_TENSOR);

  MS_LOG(DEBUG) << "create OpenCL Kernel";
  std::vector<lite::Tensor *> inputs{&input};
  std::vector<lite::Tensor *> outputs{&output};
  auto kernel = std::make_unique<PadOpenCLKernel>(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (kernel == nullptr) {
    return;
  }
  kernel->Init();

  MS_LOG(DEBUG) << "create SubGraph";
  std::vector<kernel::LiteKernel *> kernels{kernel.release()};
  auto sub_graph = new (std::nothrow) SubGraphOpenCLKernel({&input}, {&output}, kernels, kernels, kernels);
  input.MallocData(allocator);
  sub_graph->Init();
  memcpy(input.data_c(), input_data, input.Size());
  sub_graph->Run();
  if (CommonTest::CompareOutputData(reinterpret_cast<float *>(output.data_c()), const_cast<float *>(expect_data),
                                    static_cast<size_t>(output.ElementsNum()))) {
    FAIL();
  } else {
    std::cout << "COMPARE SUCCESS!\n";
  }

  MS_LOG(DEBUG) << "release resources";
  input.set_data(nullptr);
  output.set_data(nullptr);
  delete sub_graph;
}

TEST_F(TestPadOpenCL, TestPad3) {
  auto param = static_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "PadParameter create error.";
    return;
  }
  param->pad_mode_ = PaddingMode_CONSTANT;
  param->constant_value_ = 0.0f;
  param->padding_length = MAX_PAD_SIZE;
  int paddings[MAX_PAD_SIZE] = {0, 0, 3, 3, 3, 3, 0, 0};
  memcpy(param->paddings_, paddings, sizeof(paddings));

  float input_data[48] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
                          12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                          24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                          36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0};
  float expect_data[300] = {
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  36.0,
    37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0};

  TEST_MAIN(param, Format_NHWC, Format_NHWC, Format_NHWC4, kNumberTypeFloat32, {1, 4, 4, 3}, {1, 10, 10, 3}, input_data,
            expect_data);
  TEST_MAIN(param, Format_NHWC, Format_NHWC, Format_NC4HW4, kNumberTypeFloat32, {1, 4, 4, 3}, {1, 10, 10, 3},
            input_data, expect_data);
  TEST_MAIN(param, Format_NHWC, Format_NHWC, Format_NHWC4, kNumberTypeFloat16, {1, 4, 4, 3}, {1, 10, 10, 3}, input_data,
            expect_data);
  TEST_MAIN(param, Format_NHWC, Format_NHWC, Format_NC4HW4, kNumberTypeFloat16, {1, 4, 4, 3}, {1, 10, 10, 3},
            input_data, expect_data);
}

TEST_F(TestPadOpenCL, TestPad4) {
  auto param = static_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "PadParameter create error.";
    return;
  }
  param->pad_mode_ = PaddingMode_CONSTANT;
  param->constant_value_ = 1.0f;
  param->padding_length = MAX_PAD_SIZE;
  int paddings[MAX_PAD_SIZE] = {0, 0, 3, 3, 3, 3, 0, 0};
  memcpy(param->paddings_, paddings, sizeof(paddings));

  float input_data[48] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
                          12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                          24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                          36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0};
  float expect_data[300] = {
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  36.0,
    37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0};

  TEST_MAIN(param, Format_NHWC, Format_NHWC, Format_NHWC4, kNumberTypeFloat32, {1, 4, 4, 3}, {1, 10, 10, 3}, input_data,
            expect_data);
}

}  // namespace mindspore
