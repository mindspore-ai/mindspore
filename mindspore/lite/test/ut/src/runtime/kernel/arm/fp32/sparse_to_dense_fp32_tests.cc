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
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/sparse_to_dense_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

class TestSparseToDenseFp32 : public mindspore::CommonTest {
 public:
  TestSparseToDenseFp32() {}
};

TEST_F(TestSparseToDenseFp32, SparseToDense_test1) {
  std::vector<int> input1 = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  std::vector<int> shape1 = {6, 2};
  std::vector<int> input2 = {6, 10};
  std::vector<int> shape2 = {2};
  std::vector<float> input3 = {1};
  std::vector<int> shape3 = {1};
  std::vector<float> input4 = {0};
  std::vector<int> shape4 = {1};

  TypeId tid = kNumberTypeFloat32;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->set_data_type(tid);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->set_data_type(tid);

  lite::Tensor *input_tensor3 = new lite::Tensor;
  input_tensor3->set_data(input3.data());
  input_tensor3->set_shape(shape3);
  input_tensor3->set_data_type(tid);

  lite::Tensor *input_tensor4 = new lite::Tensor;
  input_tensor4->set_data(input4.data());
  input_tensor4->set_shape(shape4);
  input_tensor4->set_data_type(tid);

  std::vector<lite::Tensor *> inputs_tensor(4);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;
  inputs_tensor[2] = input_tensor3;
  inputs_tensor[3] = input_tensor4;

  const int output_size = 60;
  float output[60];
  std::vector<int> output_shape = {6, 10};

  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->set_data_type(tid);
  std::vector<lite::Tensor *> outputs_tensor(1);
  outputs_tensor[0] = output0_tensor;

  SparseToDenseParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.validate_indices_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, tid, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  input_tensor3->set_data(nullptr);
  input_tensor4->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete input_tensor3;
  delete input_tensor4;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test2) {
  std::vector<int> input1 = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  std::vector<int> shape1 = {6, 2};
  std::vector<int> input2 = {6, 10};
  std::vector<int> shape2 = {2};
  std::vector<float> input3 = {1, 2, 3, 4, 5, 6};
  std::vector<int> shape3 = {6};
  std::vector<float> input4 = {0};
  std::vector<int> shape4 = {1};

  TypeId tid = kNumberTypeFloat32;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->set_data_type(tid);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->set_data_type(tid);

  lite::Tensor *input_tensor3 = new lite::Tensor;
  input_tensor3->set_data(input3.data());
  input_tensor3->set_shape(shape3);
  input_tensor3->set_data_type(tid);

  lite::Tensor *input_tensor4 = new lite::Tensor;
  input_tensor4->set_data(input4.data());
  input_tensor4->set_shape(shape4);
  input_tensor4->set_data_type(tid);

  std::vector<lite::Tensor *> inputs_tensor(4);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;
  inputs_tensor[2] = input_tensor3;
  inputs_tensor[3] = input_tensor4;

  const int output_size = 60;
  float output[60];
  std::vector<int> output_shape = {6, 10};

  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->set_data_type(tid);
  std::vector<lite::Tensor *> outputs_tensor(1);
  outputs_tensor[0] = output0_tensor;

  SparseToDenseParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.validate_indices_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, tid, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  input_tensor3->set_data(nullptr);
  input_tensor4->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete input_tensor3;
  delete input_tensor4;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test3) {
  std::vector<int> input1 = {1, 3, 4};
  std::vector<int> shape1 = {3};
  std::vector<int> input2 = {1, 10};
  std::vector<int> shape2 = {2};
  std::vector<float> input3 = {1};
  std::vector<int> shape3 = {1};
  std::vector<float> input4 = {0};
  std::vector<int> shape4 = {1};

  TypeId tid = kNumberTypeFloat32;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->set_data_type(tid);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->set_data_type(tid);

  lite::Tensor *input_tensor3 = new lite::Tensor;
  input_tensor3->set_data(input3.data());
  input_tensor3->set_shape(shape3);
  input_tensor3->set_data_type(tid);

  lite::Tensor *input_tensor4 = new lite::Tensor;
  input_tensor4->set_data(input4.data());
  input_tensor4->set_shape(shape4);
  input_tensor4->set_data_type(tid);

  std::vector<lite::Tensor *> inputs_tensor(4);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;
  inputs_tensor[2] = input_tensor3;
  inputs_tensor[3] = input_tensor4;

  const int output_size = 10;
  float output[10];
  std::vector<int> output_shape = {1, 10};

  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->set_data_type(tid);
  std::vector<lite::Tensor *> outputs_tensor(1);
  outputs_tensor[0] = output0_tensor;

  SparseToDenseParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, tid, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<float> except_result = {0, 1, 0, 1, 1, 0, 0, 0, 0, 0};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  input_tensor3->set_data(nullptr);
  input_tensor4->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete input_tensor3;
  delete input_tensor4;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test4) {
  std::vector<int> input1 = {5};
  std::vector<int> shape1 = {1};
  std::vector<int> input2 = {10};
  std::vector<int> shape2 = {1};
  std::vector<float> input3 = {1};
  std::vector<int> shape3 = {1};
  std::vector<float> input4 = {0};
  std::vector<int> shape4 = {1};

  TypeId tid = kNumberTypeFloat32;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->set_data_type(tid);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->set_data_type(tid);

  lite::Tensor *input_tensor3 = new lite::Tensor;
  input_tensor3->set_data(input3.data());
  input_tensor3->set_shape(shape3);
  input_tensor3->set_data_type(tid);

  lite::Tensor *input_tensor4 = new lite::Tensor;
  input_tensor4->set_data(input4.data());
  input_tensor4->set_shape(shape4);
  input_tensor4->set_data_type(tid);

  std::vector<lite::Tensor *> inputs_tensor(4);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;
  inputs_tensor[2] = input_tensor3;
  inputs_tensor[3] = input_tensor4;

  const int output_size = 10;
  float output[10];
  std::vector<int> output_shape = {1, 10};

  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->set_data_type(tid);
  std::vector<lite::Tensor *> outputs_tensor(1);
  outputs_tensor[0] = output0_tensor;

  SparseToDenseParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, tid, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<float> except_result = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  input_tensor3->set_data(nullptr);
  input_tensor4->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete input_tensor3;
  delete input_tensor4;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test5) {
  std::vector<int> input1 = {0, 0, 1, 2, 2, 3, 2, 3, 4, 7, 5, 9};
  std::vector<int> shape1 = {6, 2};
  std::vector<int> input2 = {6, 10};
  std::vector<int> shape2 = {2};
  std::vector<float> input3 = {1, 2, 3, 4, 5, 6};
  std::vector<int> shape3 = {6};
  std::vector<float> input4 = {0};
  std::vector<int> shape4 = {1};

  TypeId tid = kNumberTypeFloat32;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->set_data_type(tid);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->set_data_type(tid);

  lite::Tensor *input_tensor3 = new lite::Tensor;
  input_tensor3->set_data(input3.data());
  input_tensor3->set_shape(shape3);
  input_tensor3->set_data_type(tid);

  lite::Tensor *input_tensor4 = new lite::Tensor;
  input_tensor4->set_data(input4.data());
  input_tensor4->set_shape(shape4);
  input_tensor4->set_data_type(tid);

  std::vector<lite::Tensor *> inputs_tensor(4);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;
  inputs_tensor[2] = input_tensor3;
  inputs_tensor[3] = input_tensor4;

  const int output_size = 60;
  float output[60];
  std::vector<int> output_shape = {6, 10};

  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->set_data_type(tid);
  std::vector<lite::Tensor *> outputs_tensor(1);
  outputs_tensor[0] = output0_tensor;

  SparseToDenseParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, tid, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  input_tensor3->set_data(nullptr);
  input_tensor4->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete input_tensor3;
  delete input_tensor4;
  delete output0_tensor;
  delete ctx;
}
}  // namespace mindspore
