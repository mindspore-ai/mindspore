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

#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/arithmetic.h"

namespace mindspore {

template <class T>
static void BoardcaseAdd(const T *a, const T b, T *c, const int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b;
  }
}

template <class T>
static void ElementAdd(const T *a, const T *b, T *c, const int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

template <class T>
static bool DataCompare(const T *a, const T *b, const int size, const float accuracy = 1e-4) {
  for (int i = 0; i < size; i++) {
    auto diff = fabs(a[i] - b[i]);
    if (diff > accuracy) {
      MS_LOG(ERROR) << "compare failed at " << i << " exp " << a[i] << " bug got " << b[i];
      return false;
    }
  }
  return true;
}

template <class T>
static void InitData(void *data, const int size) {
  T *data_float = reinterpret_cast<T *>(data);
  static unsigned int seed = 123;
  for (int i = 0; i < size; i++) {
    data_float[i] = static_cast<int>(rand_r(&seed)) % 100;
  }
}

template <class T>
static void LogData(void *data, const int size, const std::string prefix) {
  std::cout << prefix;
  T *data_float = reinterpret_cast<T *>(data);
  for (int i = 0; i < size; i++) {
    std::cout << data_float[i] << ",";
  }
  std::cout << std::endl;
}

template <class T>
static void TestCase(const std::vector<int> &shape_a, const std::vector<int> &shape_b) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  bool is_bias_add = shape_b.empty();
  auto data_type = kNumberTypeFloat32;
  if (sizeof(T) == 2) {
    data_type = kNumberTypeFloat16;
    ocl_runtime->SetFp16Enable(true);
  }

  lite::Tensor *tensor_a = new (std::nothrow) lite::Tensor(data_type, shape_a, schema::Format_NHWC4);
  lite::Tensor *tensor_b = new (std::nothrow) lite::Tensor(data_type, shape_b, schema::Format_NHWC4);
  lite::Tensor *tensor_c = new (std::nothrow) lite::Tensor(data_type, shape_a, schema::Format_NHWC4);
  if (tensor_a == nullptr || tensor_b == nullptr || tensor_c == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed!";
    delete tensor_a;
    delete tensor_b;
    delete tensor_c;
    return;
  }

  int64_t element_num = tensor_a->ElementsC4Num();
  int64_t element_num_b = is_bias_add ? 1 : tensor_b->ElementsC4Num();

  T *data_a = new (std::nothrow) T[element_num];
  T *data_b = new (std::nothrow) T[element_num_b];
  T *data_c_cpu = new (std::nothrow) T[element_num];
  T *data_c_ocl = new (std::nothrow) T[element_num];
  if (data_a == nullptr || data_b == nullptr || data_c_cpu == nullptr || data_c_ocl == nullptr) {
    MS_LOG(ERROR) << "Create buffer failed!";
    delete tensor_a;
    delete tensor_b;
    delete tensor_c;
    delete[] data_a;
    delete[] data_b;
    delete[] data_c_cpu;
    delete[] data_c_ocl;
    return;
  }

  InitData<T>(data_a, element_num);
  InitData<T>(data_b, element_num_b);
  memset(data_c_ocl, 0, sizeof(T) * element_num);

  if (is_bias_add) {
    BoardcaseAdd(data_a, static_cast<T *>(data_b)[0], data_c_cpu, element_num);
  } else {
    ElementAdd(data_a, data_b, data_c_cpu, element_num);
  }

  std::vector<lite::Tensor *> inputs = {tensor_a};
  if (!is_bias_add) {
    inputs.push_back(tensor_b);
  } else {
    tensor_b->MallocData();
    memcpy(tensor_b->data_c(), data_b, sizeof(T));
  }
  std::vector<lite::Tensor *> outputs = {tensor_c};

  ArithmeticParameter *param = new (std::nothrow) ArithmeticParameter();
  param->broadcasting_ = is_bias_add;
  if (param == nullptr) {
    MS_LOG(ERROR) << "Create parameter failed!";
    delete tensor_a;
    delete tensor_b;
    delete tensor_c;
    delete[] data_a;
    delete[] data_b;
    delete[] data_c_cpu;
    delete[] data_c_ocl;
    return;
  }
  param->ndim_ = 4;
  param->op_parameter_.type_ = PrimitiveType_Add;

  std::vector<lite::Tensor *> arithmetic_inputs = {tensor_a, tensor_b};
  lite::Context ctx;
  auto *arith_kernel = new (std::nothrow)
    kernel::ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(param), arithmetic_inputs, outputs, &ctx);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "Create ArithmeticOpenCLKernel failed!";
    delete tensor_a;
    delete tensor_b;
    delete tensor_c;
    delete[] data_a;
    delete[] data_b;
    delete[] data_c_cpu;
    delete[] data_c_ocl;
    delete param;
    return;
  }
  arith_kernel->Init();

  tensor_a->MallocData(allocator);
  tensor_b->MallocData(allocator);
  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto *kernel = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "Create SubGraphOpenCLKernel failed!";
    delete tensor_a;
    delete tensor_b;
    delete tensor_c;
    delete[] data_a;
    delete[] data_b;
    delete[] data_c_cpu;
    delete[] data_c_ocl;
    delete arith_kernel;
    return;
  }
  kernel->Init();

  memcpy(inputs[0]->data_c(), data_a, sizeof(T) * element_num);
  if (!is_bias_add) {
    memcpy(inputs[1]->data_c(), data_b, sizeof(T) * element_num_b);
  }

  kernel->Run();

  memcpy(data_c_ocl, outputs[0]->data_c(), sizeof(T) * element_num);

  LogData<T>(data_a, 10, "Data A : ");
  LogData<T>(data_b, tensor_b->shape().empty() ? 1 : 10, "Data B : ");
  LogData<T>(data_c_cpu, 10, "Expect compute : ");
  LogData<T>(outputs[0]->data_c(), 10, "OpenCL compute : ");
  bool cmp = DataCompare(data_c_cpu, data_c_ocl, element_num);
  MS_LOG(INFO) << "Compare " << (cmp ? "success!" : "failed!");
  EXPECT_EQ(true, cmp);

  // free
  delete[] data_a;
  delete[] data_b;
  delete[] data_c_cpu;
  delete[] data_c_ocl;

  delete kernel;
  delete arith_kernel;
  delete param;
  for (auto tensor : inputs) {
    delete tensor;
  }
  for (auto tensor : outputs) {
    delete tensor;
  }
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

class TestArithmeticOpenCL : public mindspore::CommonTest {
 public:
  TestArithmeticOpenCL() {}
};

TEST_F(TestArithmeticOpenCL, AddElementwiseFP32) {
  const std::vector<int> &shape_a = {1, 1024, 1024, 4};
  const std::vector<int> &shape_b = {1, 1024, 1024, 4};
  TestCase<float>(shape_a, shape_b);
}

TEST_F(TestArithmeticOpenCL, AddBroadcastFP32) {
  const std::vector<int> &shape_a = {1, 128, 128, 4};
  const std::vector<int> &shape_b = {};
  TestCase<float>(shape_a, shape_b);
}

TEST_F(TestArithmeticOpenCL, AddElementwiseFP16) {
  const std::vector<int> &shape_a = {1, 1024, 1024, 4};
  const std::vector<int> &shape_b = {1, 1024, 1024, 4};
  TestCase<float16_t>(shape_a, shape_b);
}

TEST_F(TestArithmeticOpenCL, AddBroadcastFP16) {
  const std::vector<int> &shape_a = {1, 128, 128, 4};
  const std::vector<int> &shape_b = {};
  TestCase<float16_t>(shape_a, shape_b);
}
}  // namespace mindspore
