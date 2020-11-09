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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/transpose.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestTransposeOpenCL : public mindspore::CommonTest {
 public:
  TestTransposeOpenCL() {}
};

void RunTestTranspose(const std::vector<int> &shape, void *input_data, void *output_data, bool enable_fp16) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto param = static_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->num_axes_ = 4;
  param->perm_[0] = shape[3];
  param->perm_[1] = shape[4];
  param->perm_[2] = shape[5];
  param->perm_[3] = shape[6];
  auto allocator = ocl_runtime->GetAllocator();
  int h = shape[0];
  int w = shape[1];
  int c = shape[2];
  std::vector<int> input_shape = {1, h, w, c};
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     input_shape, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> out_shape = {input_shape[param->perm_[0]], input_shape[param->perm_[1]],
                                input_shape[param->perm_[2]], input_shape[param->perm_[3]]};
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                       out_shape, schema::Format_NHWC);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::TransposeOpenCLKernel>(
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
  memcpy(inputs[0]->MutableData(), input_data, h * w * c * dtype_size);
  pGraph->Run();

  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, h * w * c, static_cast<float16_t>(1e-3), 2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, h * w * c, static_cast<float>(1e-5));
  }

  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test TransposeFp32 passed";
}

TEST_F(TestTransposeOpenCL, TransposeNHWC2NCHWFp32) {
  int h = 2;
  int w = 2;
  int c = 3;
  int perm0 = 0;
  int perm1 = 3;
  int perm2 = 1;
  int perm3 = 2;
  std::vector<int> shape = {h, w, c, perm0, perm1, perm2, perm3};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {0.0f, 3.0f, 6.0f, 9.0f, 1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f, 8.0f, 11.0f};

  RunTestTranspose(shape, input_data.data(), output_data.data(), false);
}

TEST_F(TestTransposeOpenCL, TransposeNHWC2NCHWFp16) {
  int h = 2;
  int w = 2;
  int c = 3;
  int perm0 = 0;
  int perm1 = 3;
  int perm2 = 1;
  int perm3 = 2;
  std::vector<int> shape = {h, w, c, perm0, perm1, perm2, perm3};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float16_t> output_data = {0.0f, 3.0f, 6.0f, 9.0f, 1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f, 8.0f, 11.0f};

  RunTestTranspose(shape, input_data.data(), output_data.data(), true);
}

TEST_F(TestTransposeOpenCL, TransposeNCHW2NHWCFp32) {
  int h = 2;
  int w = 2;
  int c = 3;
  int perm0 = 0;
  int perm1 = 2;
  int perm2 = 3;
  int perm3 = 1;
  std::vector<int> shape = {h, w, c, perm0, perm1, perm2, perm3};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> output_data = {0.0f, 6.0f, 1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f, 4.0f, 10.0f, 5.0f, 11.0f};

  RunTestTranspose(shape, input_data.data(), output_data.data(), false);
}

TEST_F(TestTransposeOpenCL, TransposeNCHW2NHWCFp16) {
  int h = 2;
  int w = 2;
  int c = 3;
  int perm0 = 0;
  int perm1 = 2;
  int perm2 = 3;
  int perm3 = 1;
  std::vector<int> shape = {h, w, c, perm0, perm1, perm2, perm3};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float16_t> output_data = {0.0f, 6.0f, 1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f, 4.0f, 10.0f, 5.0f, 11.0f};

  RunTestTranspose(shape, input_data.data(), output_data.data(), true);
}
}  // namespace mindspore
