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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/scale.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestScaleOpenCL : public mindspore::CommonTest {
 public:
  TestScaleOpenCL() {}
};

void RunTestCaseScale(void *input_data0, const std::vector<int> &input_shape, void *scale_data, void *offset_data,
                      const std::vector<int> &weight_shape, void *output_data, const std::vector<int> &out_shape,
                      bool enable_fp16, int axis, int act_type = schema::ActivationType_NO_ACTIVATION) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<ScaleParameter *>(malloc(sizeof(ScaleParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->axis_ = axis;
  param->activation_type_ = act_type;
  auto tensor_x_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), input_shape);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }

  auto tensor_scale_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), weight_shape,
                                   schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto tensor_scale = tensor_scale_ptr.get();
  if (tensor_scale == nullptr) {
    MS_LOG(ERROR) << "tensor_scale create error.";
    return;
  }
  tensor_scale->set_data(scale_data);
  auto tensor_offset_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), weight_shape,
                                   schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto tensor_offset = tensor_offset_ptr.get();
  if (tensor_offset == nullptr) {
    MS_LOG(ERROR) << "tensor_offset create error.";
    return;
  }
  tensor_offset->set_data(offset_data);
  auto tensor_out_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), out_shape);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x, tensor_scale, tensor_offset};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto op_kernel_ptr =
    std::make_unique<kernel::ScaleOpenCLKernel>(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  auto op_kernel = op_kernel_ptr.release();
  if (op_kernel == nullptr) {
    MS_LOG(ERROR) << "op_kernel create error.";
    return;
  }
  op_kernel->Init();
  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{op_kernel};

  std::vector<lite::Tensor *> inputs_g{tensor_x};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs_g, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data0, tensor_x->ElementsNum() * dtype_size);
  pGraph->Run();
  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, tensor_out->ElementsNum(), static_cast<float16_t>(1e-3),
                  2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, tensor_out->ElementsNum(), static_cast<float>(1e-5));
  }

  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }
  MS_LOG(INFO) << "TestScale passed";
}

TEST_F(TestScaleOpenCL, ScaleAxis3Fp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> weight_shape = {c};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> scale_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> offset_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {2.0f, 6.0f, 12.0f, 5.0f, 12.0f, 21.0f, 8.0f, 18.0f, 30.0f, 11.0f, 24.0f, 39.0f};
  RunTestCaseScale(input_data.data(), in_shape0, scale_data.data(), offset_data.data(), weight_shape,
                   output_data.data(), out_shape, false, 3);
}

TEST_F(TestScaleOpenCL, ScaleAxis1Fp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> weight_shape = {h};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> offset_data = {1.0f, 2.0f};
  std::vector<float> output_data = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f};
  RunTestCaseScale(input_data.data(), in_shape0, scale_data.data(), offset_data.data(), weight_shape,
                   output_data.data(), out_shape, false, 1);
}

TEST_F(TestScaleOpenCL, ScaleAxis3ReLU6Fp32) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> weight_shape = {c};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> scale_data = {1.0f, 2.0f, -1.0f};
  std::vector<float> offset_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {2.0f, 6.0f, 0.0f, 5.0f, 6.0f, 0.0f, 6.0f, 6.0f, 0.0f, 6.0f, 6.0f, 0.0f};
  RunTestCaseScale(input_data.data(), in_shape0, scale_data.data(), offset_data.data(), weight_shape,
                   output_data.data(), out_shape, false, 3, schema::ActivationType_RELU6);
}

TEST_F(TestScaleOpenCL, ScaleAxis3Fp16) {
  int n = 1;
  int h = 2;
  int w = 2;
  int c = 3;
  std::vector<int> in_shape0 = {n, h, w, c};
  std::vector<int> weight_shape = {c};
  std::vector<int> out_shape = {n, h, w, c};
  std::vector<float16_t> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float16_t> scale_data = {1.0f, 2.0f, 3.0f};
  std::vector<float16_t> offset_data = {1.0f, 2.0f, 3.0f};
  std::vector<float16_t> output_data = {2.0f, 6.0f, 12.0f, 5.0f, 12.0f, 21.0f, 8.0f, 18.0f, 30.0f, 11.0f, 24.0f, 39.0f};
  RunTestCaseScale(input_data.data(), in_shape0, scale_data.data(), offset_data.data(), weight_shape,
                   output_data.data(), out_shape, true, 3);
}
}  // namespace mindspore
