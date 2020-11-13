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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/one_hot.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestOneHotOpenCL : public mindspore::CommonTest {
 public:
  TestOneHotOpenCL() {}
};

void RunTestCaseOneHot(const std::vector<int> &shape_in, const std::vector<int> &shape_out, void *input_data,
                       void *output_data, int axis, int depth, float on_value, float off_value) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->axis_ = axis;
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(kNumberTypeFloat32, shape_in, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  std::vector<int> weight_shape = {};
  auto tensor_depth_ptr = std::make_unique<lite::Tensor>(kNumberTypeInt32, weight_shape, schema::Format_NHWC);
  auto tensor_depth = tensor_depth_ptr.get();
  if (tensor_depth == nullptr) {
    MS_LOG(ERROR) << "tensor_depth create error.";
    return;
  }
  tensor_depth->set_data(&depth);
  auto tensor_on_value_ptr = std::make_unique<lite::Tensor>(kNumberTypeFloat32, weight_shape, schema::Format_NHWC);
  auto tensor_on_value = tensor_on_value_ptr.get();
  if (tensor_on_value == nullptr) {
    MS_LOG(ERROR) << "tensor_on_value create error.";
    return;
  }
  tensor_on_value->set_data(&on_value);
  auto tensor_off_value_ptr = std::make_unique<lite::Tensor>(kNumberTypeFloat32, weight_shape, schema::Format_NHWC);
  auto tensor_off_value = tensor_off_value_ptr.get();
  if (tensor_off_value == nullptr) {
    MS_LOG(ERROR) << "tensor_off_value create error.";
    return;
  }
  tensor_off_value->set_data(&off_value);
  auto tensor_out_ptr = std::make_unique<lite::Tensor>(kNumberTypeFloat32, shape_out);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x, tensor_depth, tensor_on_value, tensor_off_value};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::OneHotOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(param), nullptr, kernel::KernelKey(), nullptr);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  std::vector<lite::Tensor *> inputs_g{tensor_x};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs_g, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data, inputs[0]->ElementsNum() * sizeof(int));
  pGraph->Run();

  CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-5));
  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test OneHot passed";
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis3Fp32) {
  int depth = 4;
  int axis = -1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {3, 4, -1, 2};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis3T2Fp32) {
  int depth = 5;
  int axis = -1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {-1, 3, 4, 5};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis3T3Fp32) {
  int depth = 9;
  int axis = -1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 3};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4, 9, 8, 9, 1, 8};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis3T4Fp32) {
  int depth = 6;
  int axis = -1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {2, 4, 0, 6, 1, 6, 2, 2, 4, 5};
  std::vector<float> output_data = {-1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f,
                                    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 2.f;
  float off_value = 0.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {2, 3, 0, 3};
  std::vector<float> output_data = {0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
                                    2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis2T2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 2.f;
  float off_value = 0.f;
  std::vector<int> shape_in = {1, 6, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {1, 1, 1, 0, 1, 1, 4, -1, 4, 4, -1, 1};
  std::vector<float> output_data = {0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f,
                                    2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis2T3Fp32) {
  int depth = 1;
  int axis = 2;
  float on_value = 2.f;
  float off_value = 0.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {-1, 1, -1, 0};
  std::vector<float> output_data = {0.0f, 0.0f, 0.0f, 2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis2T4Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4, 0, -1, 2, 5, 4, -1, 4, 4, 4};
  std::vector<float> output_data = {-1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis1T1Fp32) {
  int depth = 1;
  int axis = 1;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {1, 6, 6};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {0,  -1, 1, 0, -1, -1, 0, 0,  -1, 1, 0, -1, -1, 1, 1, -1, 1, 1,
                                 -1, 1,  1, 1, -1, 0,  0, -1, 0,  0, 1, 1,  1,  1, 0, 0,  0, -1};
  std::vector<float> output_data = {2.0f,  -2.0f, -2.0f, 2.0f,  -2.0f, -2.0f, 2.0f,  2.0f,  -2.0f, -2.0f, 2.0f,  -2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,
                                    2.0f,  -2.0f, 2.0f,  2.0f,  -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  2.0f,  2.0f,  -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis1T2Fp32) {
  int depth = 4;
  int axis = 1;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {-1, 1, 1, 2};
  std::vector<float> output_data = {-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  2.0f,  -2.0f,
                                    -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f, -2.0f, -2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis1T3Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {3, 5, 2, 0, 2, 2, -1, 0, 4, 3};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                                    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis0Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {1, 2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4, 0, 3, 3};
  std::vector<float> output_data = {-2.0f, 2.0f,  -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  2.0f,  2.0f,  -2.0f, -2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {1, 2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {2, 4, 4, 3, 5, 0, 3, 3, -1, 2};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
                                    -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f,
                                    -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot4DAxis0T3Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {2, 2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {0, 3, 2, 0, 0, 3, 4, 1, 5, 1, 4, -1, 3, 3, 1, 1, 4, 2, 2, 4};
  std::vector<float> output_data = {
    1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f,
    -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f,
    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis0Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {2, 3};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4, 4, 3, 2, -1, 5};
  std::vector<float> output_data = {-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f, -2.0f, -2.0f, -2.0f,
                                    2.0f,  -2.0f, -2.0f, -2.0f, 2.0f,  2.0f,  -2.0f, -2.0f, -2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4, 2, 2, 3, -1, 5, 2, 4, 5, -1};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis1Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {2, 3};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {0, 0, 0, 0, 4, -1};
  std::vector<float> output_data = {2.0f,  2.0f,  2.0f,  -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f, -2.0f, -2.0f, -2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis1T2Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {1, -1, 3, 2, 5, 5, 4, 5, 0, -1};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis2Fp32) {
  int depth = 4;
  int axis = 2;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {2, 2};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {0, 3, 4, 2};
  std::vector<float> output_data = {2.0f,  -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot3DAxis2T2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {2, 5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {0, -1, 2, -1, 5, 4, 2, -1, 4, -1};
  std::vector<float> output_data = {1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
                                    -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot2DAxis0Fp32) {
  int depth = 3;
  int axis = 0;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {3};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {2, 1, 3};
  std::vector<float> output_data = {-2.0f, -2.0f, -2.0f, -2.0f, 2.0f, -2.0f, 2.0f, -2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot2DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {2, 2, 0, 0, 4};
  std::vector<float> output_data = {-1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot2DAxis1Fp32) {
  int depth = 3;
  int axis = -1;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {3};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {1, 2, 0};
  std::vector<float> output_data = {-2.0f, 2.0f, -2.0f, -2.0f, -2.0f, 2.0f, 2.0f, -2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot2DAxis1T2Fp32) {
  int depth = 5;
  int axis = -1;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {5};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {5, 4, 0, 4, -1};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot1DAxis0Fp32) {
  int depth = 3;
  int axis = -1;
  float on_value = 2.f;
  float off_value = -2.f;
  std::vector<int> shape_in = {};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {1};
  std::vector<float> output_data = {-2.0f, 2.0f, -2.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}

TEST_F(TestOneHotOpenCL, OneHot1DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1.f;
  float off_value = -1.f;
  std::vector<int> shape_in = {};
  std::vector<int> shape_out = shape_in;
  shape_out.insert(shape_out.begin() + (axis + shape_in.size() + 1) % (shape_in.size() + 1), depth);
  std::vector<int> input_data = {4};
  std::vector<float> output_data = {-1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

  RunTestCaseOneHot(shape_in, shape_out, input_data.data(), output_data.data(), axis, depth, on_value, off_value);
}
}  // namespace mindspore
