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
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/power.h"

using mindspore::lite::Tensor;
using mindspore::schema::Format::Format_NHWC;
namespace mindspore {
class TestPowerOpenCLCI : public mindspore::CommonTest {
 public:
  TestPowerOpenCLCI() {}
};
template <class T>
void CompareData(const T *output_data, const T *correct_data, int size, float err_bound) {
  for (int i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}
template <class T>
void TEST_MAIN(const T *input_data1, const T *input_data2, const T *expect_data, const TypeId data_type,
               const std::vector<int> &shape_a, const std::vector<int> &shape_b, const std::vector<int> &out_shape,
               bool broadcast, const T scale = 1.0, const T shift = 0, const T exponent = 2) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  if (data_type == kNumberTypeFloat16) {
    runtime->SetFp16Enable(true);
  }
  auto allocator = runtime->GetAllocator();
  auto tensor_type = lite::Tensor::CONST_TENSOR;

  auto in_tensor1 = Tensor(data_type, shape_a, Format_NHWC, tensor_type);
  auto in_tensor2 = Tensor(data_type, shape_b, Format_NHWC, tensor_type);
  auto output_tensor = Tensor(data_type, out_shape, Format_NHWC, tensor_type);

  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }
  param->scale_ = scale;
  param->shift_ = shift;
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs{&output_tensor};
  if (broadcast) {
    param->broadcast_ = true;
    inputs.push_back(&in_tensor1);
    param->power_ = exponent;
  } else {
    inputs.push_back(&in_tensor1);
    inputs.push_back(&in_tensor2);
  }
  auto *power_kernel =
    new (std::nothrow) kernel::PowerOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (power_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::PowerOpenCLKernel failed ";
    delete param;
    return;
  }
  power_kernel->Init();
  // to do allocate memory for inputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{power_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete power_kernel;
    return;
  }
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  size_t size = 1 * sizeof(T);
  for (int i = 0; i < out_shape.size(); ++i) {
    size *= out_shape[i];
  }
  if (broadcast) {
    memcpy(inputs[0]->data_c(), input_data1, size);
  } else {
    memcpy(inputs[0]->data_c(), input_data1, size);
    memcpy(inputs[1]->data_c(), input_data2, size);
  }
  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  T *output_data_gpu = reinterpret_cast<T *>(output_tensor.data_c());
  CompareData(output_data_gpu, expect_data, output_tensor.ElementsNum(), 0.0001);
  delete sub_graph;
}

TEST_F(TestPowerOpenCLCI, Int32CI) {
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape_a = {1, 2, 8};
  std::vector<int> shape_b = {1, 2, 8};
  std::vector<int> output_shape = {1, 2, 8};
  auto data_type = kNumberTypeFloat32;
  const float input_data1[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};
  const float input_data2[] = {2, 2, 2, 1, 2, 2, 3, 3, 2, 2, 3, 0, 2, 2, 1, 2};
  const float expect_data[] = {4.0,   9.0,   16.0,   5.0, 36.0,  49.0,  512,  729,
                               100.0, 121.0, 1728.0, 1.0, 196.0, 225.0, 16.0, 289.0};
  TEST_MAIN(input_data1, input_data2, expect_data, data_type, shape_a, shape_b, output_shape, false);
}

TEST_F(TestPowerOpenCLCI, Fp32CI) {
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape_a = {2, 8};
  std::vector<int> shape_b = {2, 8};
  std::vector<int> output_shape = {2, 8};
  auto data_type = kNumberTypeFloat32;
  const float input_data1[] = {0.78957046,  -0.99770847, 1.05838929,  1.60738329,  -1.66226552, -2.03170525,
                               -0.48257631, -0.94244638, 1.47462044,  -0.80247114, 0.12354778,  -0.36436107,
                               -2.41973013, -0.40221205, -0.26739485, 0.23298305};
  const float input_data2[] = {3, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2};
  const float expect_data[] = {0.49223521, 0.99542219, 1.12018788, 1.60738329, 2.76312667, 4.1278262,
                               0.23287989, 0.88820518, 3.20657016, 0.64395994, 0.01526405, 0.13275899,
                               5.85509388, 0.16177453, 0.07150001, 0.0542811};
  TEST_MAIN(input_data1, input_data2, expect_data, data_type, shape_a, shape_b, output_shape, false);
}

TEST_F(TestPowerOpenCLCI, Fp16CI) {
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape_a = {2, 8};
  std::vector<int> shape_b = {2, 8};
  std::vector<int> output_shape = {2, 8};
  auto data_type = kNumberTypeFloat16;
  const float16_t input_data1[] = {0.1531, -0.8003, -0.1848, 0.3833, -1.469, 0.5586, -0.3223, -0.8887,
                                   0.697,  -1.007,  -0.45,   -1.736, -0.462, -0.699, -0.596,  0.7466};
  const float16_t input_data2[] = {2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0};
  const float16_t expect_data[] = {0.02344, -0.8003, -0.1848, 0.147,  2.156,  0.312, 0.1039, 0.7896,
                                   0.4856,  1.014,   0.2025,  -1.736, 0.2134, 0.489, -0.596, 0.7466};
  TEST_MAIN(input_data1, input_data2, expect_data, data_type, shape_a, shape_b, output_shape, false);
}

TEST_F(TestPowerOpenCLCI, broadcast) {
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> shape_a = {1, 2, 8};
  std::vector<int> shape_b = {};
  std::vector<int> output_shape = {1, 2, 8};
  auto data_type = kNumberTypeFloat32;
  float input_data1[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};
  float expect_data[] = {4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64, 81, 100.0, 121.0, 144, 169, 196.0, 225.0, 256, 289.0};
  TEST_MAIN(input_data1, input_data1, expect_data, data_type, shape_a, shape_b, output_shape, true);
}

}  // namespace mindspore
