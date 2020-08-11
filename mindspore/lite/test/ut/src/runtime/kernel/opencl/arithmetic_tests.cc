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

void BoardcaseAdd(const float *a, const float b, float *c, const int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b;
  }
}

void ElementAdd(const float *a, const float *b, float *c, const int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

bool DataCompare(const float *a, const float *b, const int size, const float accuracy = 1e-4) {
  for (int i = 0; i < size; i++) {
    auto diff = fabs(a[i] - b[i]);
    if (diff > accuracy) {
      MS_LOG(ERROR) << "compare failed at " << i << " exp " << a[i] << " bug got " << b[i];
      return false;
    }
  }
  return true;
}

void InitData(void *data, const int size) {
  float *data_float = reinterpret_cast<float *>(data);
  static unsigned int seed = 123;
  for (int i = 0; i < size; i++) {
    data_float[i] = static_cast<int>(rand_r(&seed)) % 100;
  }
}

void LogData(void *data, const int size, const std::string prefix) {
  std::cout << prefix;
  float *data_float = reinterpret_cast<float *>(data);
  for (int i = 0; i < size; i++) {
    std::cout << data_float[i] << ",";
  }
  std::cout << std::endl;
}

void TestCase(const std::vector<int> &shape_a, const std::vector<int> &shape_b) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  bool is_bias_add = shape_b.empty();
  auto tensorType = schema::NodeType_ValueNode;

  lite::tensor::Tensor *tensor_a =
    new lite::tensor::Tensor(kNumberTypeFloat32, shape_a, schema::Format_NHWC4, tensorType);
  lite::tensor::Tensor *tensor_b =
    new lite::tensor::Tensor(kNumberTypeFloat32, shape_b, schema::Format_NHWC4, tensorType);
  lite::tensor::Tensor *tensor_c =
    new lite::tensor::Tensor(kNumberTypeFloat32, shape_a, schema::Format_NHWC4, tensorType);
  int64_t element_num = tensor_a->ElementsC4Num();
  int64_t element_num_b = is_bias_add ? 1 : tensor_b->ElementsC4Num();

  float *data_a = new float[element_num];
  float *data_b = new float[element_num_b];
  float *data_c_cpu = new float[element_num];
  float *data_c_ocl = new float[element_num];

  InitData(data_a, element_num);
  InitData(data_b, element_num_b);
  memset(data_c_ocl, 0, sizeof(float) * element_num);

  if (is_bias_add) {
    BoardcaseAdd(data_a, static_cast<float *>(data_b)[0], data_c_cpu, element_num);
  } else {
    ElementAdd(data_a, data_b, data_c_cpu, element_num);
  }

  std::vector<lite::tensor::Tensor *> inputs = {tensor_a};
  if (!is_bias_add) {
    inputs.push_back(tensor_b);
  } else {
    tensor_b->MallocData();
    memcpy(tensor_b->Data(), data_b, sizeof(float));
  }
  std::vector<lite::tensor::Tensor *> outputs = {tensor_c};

  ArithmeticParameter *param = new ArithmeticParameter();
  param->ndim_ = 4;
  param->op_parameter_.type_ = PrimitiveType_Add;

  std::vector<lite::tensor::Tensor *> arithmetic_inputs = {tensor_a, tensor_b};
  lite::Context ctx;
  auto *arith_kernel =
    new kernel::ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(param), arithmetic_inputs, outputs, &ctx);
  arith_kernel->Init();

  tensor_a->MallocData(allocator);
  tensor_b->MallocData(allocator);
  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto *kernel = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  kernel->Init();

  memcpy(inputs[0]->Data(), data_a, sizeof(float) * element_num);
  if (!is_bias_add) {
    memcpy(inputs[1]->Data(), data_b, sizeof(float) * element_num_b);
  }

  kernel->Run();

  memcpy(data_c_ocl, outputs[0]->Data(), sizeof(float) * element_num);

  // ocl_runtime->SyncCommandQueue();
  LogData(data_a, 10, "Data A : ");
  LogData(data_b, tensor_b->shape().empty() ? 1 : 10, "Data B : ");
  LogData(data_c_cpu, 10, "Expect compute : ");
  LogData(outputs[0]->Data(), 10, "OpenCL compute : ");
  bool cmp = DataCompare(data_c_cpu, data_c_ocl, element_num);
  MS_LOG(INFO) << "Compare " << (cmp ? "success!" : "failed!");

  // free
  delete[] data_a;
  delete[] data_b;
  delete[] data_c_cpu;
  delete[] data_c_ocl;

  delete kernel;
  delete arith_kernel;

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

TEST_F(TestArithmeticOpenCL, AddElementwiseTest) {
  const std::vector<int> &shape_a = {1, 1024, 1024, 4};
  const std::vector<int> &shape_b = {1, 1024, 1024, 4};
  TestCase(shape_a, shape_b);
}

TEST_F(TestArithmeticOpenCL, AddBoardcaseTest) {
  const std::vector<int> &shape_a = {1, 128, 128, 4};
  const std::vector<int> &shape_b = {};
  TestCase(shape_a, shape_b);
}

}  // namespace mindspore
