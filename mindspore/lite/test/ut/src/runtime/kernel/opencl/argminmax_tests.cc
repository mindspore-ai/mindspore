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
#include "src/runtime/kernel/opencl/utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/argminmax.h"

namespace mindspore {
class TestArgMinMaxOpenCL : public mindspore::CommonTest {
 public:
  TestArgMinMaxOpenCL() {}
};
template <typename T>
void test_main_argminmax(void *input_data, void *correct_data, const std::vector<int> &input_shape,
                         const std::vector<int> &output_shape, ArgMinMaxParameter *param, TypeId data_type,
                         schema::Format format) {
  MS_LOG(INFO) << " begin test ";
  auto ocl_runtime_wrap = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = ocl_runtime_wrap.GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  auto tensor_a = lite::Tensor(TypeId(data_type), input_shape, format);
  auto tensor_c = lite::Tensor(TypeId(data_type), output_shape, format);
  std::vector<lite::Tensor *> inputs{&tensor_a};
  std::vector<lite::Tensor *> outputs{&tensor_c};
  size_t input_size = tensor_a.Size();

  auto *pkernel =
    new (std::nothrow) kernel::ArgMinMaxOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (pkernel == nullptr) {
    MS_LOG(INFO) << "new SpaceToBatchNDOpenCLKernel failed ";
    return;
  }
  pkernel->Init();

  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }

  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{pkernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    delete pkernel;
    MS_LOG(INFO) << " new SubGraphOpenCLKernel failed ";
    return;
  }
  sub_graph->Init();

  MS_LOG(INFO) << " init tensors ";
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->MutableData());
  memcpy(input_ptr, input_data, input_size);
  std::cout << "==================input data================" << std::endl;
  for (auto i = 0; i < inputs[0]->ElementsNum(); ++i) {
    std::cout << input_ptr[i] << ", ";
  }
  std::cout << std::endl;

  sub_graph->Run();

  auto *output_data = reinterpret_cast<T *>(outputs[0]->MutableData());
  std::cout << "==================output data================" << std::endl;
  for (auto i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "==================correct data================" << std::endl;
  for (auto i = 0; i < outputs[0]->ElementsNum(); ++i) {
    std::cout << static_cast<T *>(correct_data)[i] << ", ";
  }
  std::cout << std::endl;
  CommonTest::CompareOutputData<T>(output_data, static_cast<T *>(correct_data), outputs[0]->ElementsNum(), 0.0001);
  delete sub_graph;
}
TEST_F(TestArgMinMaxOpenCL, axis0topk2index) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {100, 2,  4,  50, 11, 12, 34, 35, 10, 20, 40, 5,
                                7,   80, 10, 11, 55, 25, 5,  15, 18, 8,  15, 16};
  std::vector<float> except_out = {0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 2, 2};
  param->dims_size_ = 4;
  param->axis_ = 0;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = false;
  std::vector<int> in_shape = {3, 2, 2, 2};
  std::vector<int> out_shape = {2, 2, 2, 2};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis0topk2value) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {100, 2,  4,  50, 11, 12, 34, 35, 10, 20, 40, 5,
                                7,   80, 10, 11, 55, 25, 5,  15, 18, 8,  15, 16};
  std::vector<float> except_out = {100, 25, 40, 50, 18, 80, 34, 35, 55, 20, 5, 15, 11, 12, 15, 16};
  param->dims_size_ = 4;
  param->axis_ = 0;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = true;
  std::vector<int> in_shape = {3, 2, 2, 2};
  std::vector<int> out_shape = {2, 2, 2, 2};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis1topk2index) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                                10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  std::vector<float> except_out = {0, 1, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0};
  param->dims_size_ = 4;
  param->axis_ = 1;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = false;
  std::vector<int> in_shape = {2, 3, 2, 3};
  std::vector<int> out_shape = {2, 2, 2, 3};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis1topk2value) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                                10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  std::vector<float> except_out = {100, 12, 200, 34, 50, 36,  11, 6,  17, 10, 35, 30,
                                   18,  80, 90,  40, 22, 120, 10, 20, 30, 10, 11, 60};
  param->dims_size_ = 4;
  param->axis_ = 1;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = true;
  std::vector<int> in_shape = {2, 3, 2, 3};
  std::vector<int> out_shape = {2, 2, 2, 3};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis2topk1index) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  param->dims_size_ = 4;
  param->axis_ = 2;
  param->topk_ = 1;
  param->get_max_ = true;
  param->out_value_ = false;
  std::vector<float> in_data = {10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                                10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                                10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12};
  std::vector<float> except_out = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  std::vector<int> in_shape = {2, 3, 3, 3};
  std::vector<int> out_shape = {2, 3, 1, 3};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis2topk2value) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                                20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                                30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50,
                                   30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50};
  param->dims_size_ = 4;
  param->axis_ = 2;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = true;
  std::vector<int> in_shape = {2, 2, 3, 5};
  std::vector<int> out_shape = {1, 2, 2, 5};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis2topk2index) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                                20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                                30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1,
                                   2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1};
  param->dims_size_ = 4;
  param->axis_ = 2;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = false;
  std::vector<int> in_shape = {2, 2, 3, 5};
  std::vector<int> out_shape = {2, 2, 2, 5};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis3topk2index) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                                20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                                30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1};
  param->dims_size_ = 4;
  param->axis_ = 3;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = false;
  std::vector<int> in_shape = {2, 2, 3, 5};
  std::vector<int> out_shape = {2, 2, 3, 2};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
TEST_F(TestArgMinMaxOpenCL, axis3topk2value) {
  ArgMinMaxParameter *param = std::make_unique<ArgMinMaxParameter>().release();
  if (param == nullptr) {
    return;
  }
  std::vector<float> in_data = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                                20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                                30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45,
                                   90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45};
  param->dims_size_ = 4;
  param->axis_ = 3;
  param->topk_ = 2;
  param->get_max_ = true;
  param->out_value_ = true;
  std::vector<int> in_shape = {2, 2, 3, 5};
  std::vector<int> out_shape = {2, 2, 3, 2};

  TypeId data_type = kNumberTypeFloat32;
  schema::Format format = schema::Format_NHWC;
  test_main_argminmax<float>(in_data.data(), except_out.data(), in_shape, out_shape, param, data_type, format);
}
}  // namespace mindspore
