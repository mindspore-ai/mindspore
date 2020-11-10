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
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/common/log_adapter.h"
#include "src/runtime/kernel/opencl/kernel/resize.h"
#include "src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestResizeOpenCL : public mindspore::CommonTest {
 public:
  TestResizeOpenCL() {}
};

void RunTestCaseResize(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16,
                       int resize_mode, bool align_corners) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<ResizeParameter *>(malloc(sizeof(ResizeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  int n = shape[0];
  int h = shape[1];
  int w = shape[2];
  int oh = shape[3];
  int ow = shape[4];
  int c = shape[5];
  param->new_height_ = oh;
  param->new_width_ = ow;
  param->align_corners_ = align_corners;
  param->method_ = resize_mode;
  std::vector<int> input_shape = {n, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     input_shape, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {n, oh, ow, c};
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       out_shape, schema::Format_NHWC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::ResizeOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(param), nullptr, kernel::KernelKey(), nullptr);
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

  MS_LOG(INFO) << "Test Resize passed";
}

TEST_F(TestResizeOpenCL, ResizeBilinearFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int c = 1;
  bool align_corners = false;
  std::vector<int> shape = {n, h, w, oh, ow, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.5f, 2.0f, 2.0f,
                                    2.0f, 2.5f, 3.0f, 3.0f, 2.0f, 2.5f, 3.0f, 3.0f};
  RunTestCaseResize(shape, input_data.data(), output_data.data(), false, schema::ResizeMethod_LINEAR, align_corners);
}

TEST_F(TestResizeOpenCL, ResizeBilinearFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int c = 1;
  bool align_corners = false;
  std::vector<int> shape = {n, h, w, oh, ow, c};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float16_t> output_data = {0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.5f, 2.0f, 2.0f,
                                        2.0f, 2.5f, 3.0f, 3.0f, 2.0f, 2.5f, 3.0f, 3.0f};
  RunTestCaseResize(shape, input_data.data(), output_data.data(), true, schema::ResizeMethod_LINEAR, align_corners);
}

TEST_F(TestResizeOpenCL, ResizeBilinearAlignFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 3;
  int ow = 3;
  int c = 1;
  bool align_corners = true;
  std::vector<int> shape = {n, h, w, oh, ow, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {0.0f, 0.5f, 1.0f, 1.0f, 1.5f, 2.0f, 2.0f, 2.5f, 3.0f};
  RunTestCaseResize(shape, input_data.data(), output_data.data(), false, schema::ResizeMethod_LINEAR, align_corners);
}

TEST_F(TestResizeOpenCL, ResizeNearestNeighborFp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int c = 1;
  bool align_corners = false;
  std::vector<int> shape = {n, h, w, oh, ow, c};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                                    2.0f, 2.0f, 3.0f, 3.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  RunTestCaseResize(shape, input_data.data(), output_data.data(), false, schema::ResizeMethod_NEAREST, align_corners);
}

TEST_F(TestResizeOpenCL, ResizeNearestNeighborFp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int c = 1;
  bool align_corners = false;
  std::vector<int> shape = {n, h, w, oh, ow, c};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float16_t> output_data = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                                        2.0f, 2.0f, 3.0f, 3.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  RunTestCaseResize(shape, input_data.data(), output_data.data(), true, schema::ResizeMethod_NEAREST, align_corners);
}
}  // namespace mindspore
