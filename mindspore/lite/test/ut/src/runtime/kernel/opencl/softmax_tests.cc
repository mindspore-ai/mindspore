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
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/softmax.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestSoftmaxOpenCL : public mindspore::CommonTest {
 public:
  TestSoftmaxOpenCL() {}
};

void RunTestCaseSoftmax(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16,
                        int axis) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  int n, h, w, c;
  bool is_2d = false;
  if (shape.size() == 2) {
    is_2d = true;
    h = w = 1;
    n = shape[0];
    c = shape[1];
  } else {
    n = shape[0];
    h = shape[1];
    w = shape[2];
    c = shape[3];
  }
  std::vector<int> input_shape = {n, h, w, c};
  if (is_2d) {
    input_shape = {n, c};
  }
  auto input_format = is_2d ? schema::Format_NC : schema::Format_NHWC;
  auto input_dtype = enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32;
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(input_dtype), input_shape, input_format);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(input_dtype), input_shape, input_format);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto opParameter = static_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter create error.";
    return;
  }
  opParameter->axis_ = axis;
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::SoftmaxOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(opParameter), nullptr, kernel::KernelKey(), nullptr);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data, inputs[0]->ElementsNum() * dtype_size);
  pGraph->Run();

  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float16_t>(1e-3),
                  2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-5));
  }
  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test Softmax passed";
}

TEST_F(TestSoftmaxOpenCL, Softmax2DFp32) {
  int n = 1;
  int c = 10;
  std::vector<int> shape = {n, c};
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output_data = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

  RunTestCaseSoftmax(shape, input_data.data(), output_data.data(), false, 1);
}

TEST_F(TestSoftmaxOpenCL, Softmax2DFp16) {
  int n = 1;
  int c = 10;
  std::vector<int> shape = {n, c};
  std::vector<float16_t> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float16_t> output_data = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

  RunTestCaseSoftmax(shape, input_data.data(), output_data.data(), true, 1);
}

TEST_F(TestSoftmaxOpenCL, Softmax4DFp32) {
  int n = 1;
  int h = 2;
  int w = 1;
  int c = 5;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output_data = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

  RunTestCaseSoftmax(shape, input_data.data(), output_data.data(), false, 3);
}

TEST_F(TestSoftmaxOpenCL, Softmax4DFp16) {
  int n = 1;
  int h = 2;
  int w = 1;
  int c = 5;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float16_t> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float16_t> output_data = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

  RunTestCaseSoftmax(shape, input_data.data(), output_data.data(), true, 3);
}

TEST_F(TestSoftmaxOpenCL, Softmax4DAxis1Fp32) {
  int n = 1;
  int h = 2;
  int w = 1;
  int c = 1;
  std::vector<int> shape = {n, h, w, c};
  std::vector<float> input_data = {1.0f, 1.0f};
  std::vector<float> output_data = {0.5f, 0.5f};

  RunTestCaseSoftmax(shape, input_data.data(), output_data.data(), false, 1);
}
}  // namespace mindspore
