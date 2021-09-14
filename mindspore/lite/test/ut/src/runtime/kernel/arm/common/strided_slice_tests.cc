/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include "common/common_test.h"
#include "nnacl/fp32/strided_slice_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "nnacl/strided_slice_parameter.h"

namespace mindspore {
class TestStridedSlice : public mindspore::CommonTest {
 public:
  TestStridedSlice() {}
};

void InitStridedSliceParam(StridedSliceParameter *param, const lite::Tensor *in_tensor,
                           const lite::Tensor *begin_tensor, const lite::Tensor *end_tensor,
                           const lite::Tensor *stride_tensor) {
  int dim = begin_tensor->ElementsNum();
  auto input_shape = in_tensor->shape();
  int *begin = reinterpret_cast<int *>(begin_tensor->data());
  int *end = reinterpret_cast<int *>(end_tensor->data());
  int *stride = reinterpret_cast<int *>(stride_tensor->data());
  for (int i = 0; i < dim; ++i) {
    param->begins_[i] = begin[i];
    param->ends_[i] = end[i];
    param->strides_[i] = stride[i];
    param->in_shape_[i] = input_shape[i];
  }
  param->num_axes_ = dim;
  param->in_shape_length_ = dim;
  param->data_type = kDataTypeFloat;
  param->begins_mask_ = 0;
  param->ends_mask_ = 0;
  param->ellipsisMask_ = 0;
  param->newAxisMask_ = 0;
  param->shrinkAxisMask_ = 0;
}

TEST_F(TestStridedSlice, StridedSlice) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {1, 2, 4});
  lite::Tensor out_tensor(kNumberTypeFloat32, {1, 1, 2});
  float input_data[] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};
  float output_data[2] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {3});
  int begins_data[] = {0, 0, 0};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {3});
  int ends_data[] = {1, 2, 4};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {3});
  int strides_data[] = {1, 2, 2};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  float expect[2] = {0.2390374, 0.05051243};
  ASSERT_NEAR(output_data[0], expect[0], 0.001);
  ASSERT_NEAR(output_data[1], expect[1], 0.001);
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

// 7d
TEST_F(TestStridedSlice, 7d) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {1, 2, 4, 1, 1, 1, 1});
  lite::Tensor out_tensor(kNumberTypeFloat32, {1, 1, 2, 1, 1, 1, 1});
  float input_data[] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};
  float output_data[2] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {7});
  int begins_data[] = {0, 0, 1, 0, 0, 0, 0};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {7});
  int ends_data[] = {1, 2, 4, 1, 1, 1, 1};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {7});
  int strides_data[] = {1, 2, 2, 1, 1, 1, 1};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  float expect[2] = {0.92039955, 0.49574447};
  ASSERT_NEAR(output_data[0], expect[0], 0.001);
  ASSERT_NEAR(output_data[1], expect[1], 0.001);
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

// 8d
TEST_F(TestStridedSlice, 8d) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 2, 2, 3, 1, 1, 1, 1});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 2, 1, 1, 1, 1});
  int8_t input_data[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int8_t output_data[2] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {8});
  int begins_data[] = {0, 0, 1, 0, 0, 0, 0, 0};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {8});
  int ends_data[] = {1, 2, 2, 3, 1, 1, 1, 1};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {8});
  int strides_data[] = {1, 2, 1, 2, 1, 1, 1, 1};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  int8_t expect[4] = {-9, -7};
  for (unsigned int i = 0; i < sizeof(expect); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

// fast run (7d)
TEST_F(TestStridedSlice, FastRun7d) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {1, 2, 4, 1, 1, 1, 1});
  lite::Tensor out_tensor(kNumberTypeFloat32, {1, 2, 2, 1, 1, 1, 1});
  float input_data[] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};
  float output_data[4] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {7});
  int begins_data[] = {0, 0, 1, 0, 0, 0, 0};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {7});
  int ends_data[] = {1, 2, 4, 1, 1, 1, 1};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {7});
  int strides_data[] = {1, 1, 2, 1, 1, 1, 1};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  float expect[4] = {0.92039955, 0.49574447, 0.02647042, 0.4566604};
  ASSERT_NEAR(output_data[0], expect[0], 0.001);
  ASSERT_NEAR(output_data[1], expect[1], 0.001);
  ASSERT_NEAR(output_data[2], expect[2], 0.001);
  ASSERT_NEAR(output_data[3], expect[3], 0.001);
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

// fast run (7d single thread)
TEST_F(TestStridedSlice, FastRun7dSingleThread) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {1, 2, 4, 1, 1, 1, 1});
  lite::Tensor out_tensor(kNumberTypeFloat32, {1, 2, 2, 1, 1, 1, 1});
  float input_data[] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};
  float output_data[4] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {7});
  int begins_data[] = {0, 0, 1, 0, 0, 0, 0};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {7});
  int ends_data[] = {1, 2, 4, 1, 1, 1, 1};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {7});
  int strides_data[] = {1, 1, 2, 1, 1, 1, 1};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  float expect[4] = {0.92039955, 0.49574447, 0.02647042, 0.4566604};
  ASSERT_NEAR(output_data[0], expect[0], 0.001);
  ASSERT_NEAR(output_data[1], expect[1], 0.001);
  ASSERT_NEAR(output_data[2], expect[2], 0.001);
  ASSERT_NEAR(output_data[3], expect[3], 0.001);
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestStridedSlice, StridedSliceInt8) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 3, 4});
  lite::Tensor out_tensor(kNumberTypeInt8, {2, 3, 4});
  int8_t input_data[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int8_t output_data[4] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begins_tensor(kNumberTypeInt32, {3});
  int begins_data[] = {0, 1, 2};
  begins_tensor.set_data(begins_data);
  lite::Tensor ends_tensor(kNumberTypeInt32, {3});
  int ends_data[] = {2, 3, 4};
  ends_tensor.set_data(ends_data);
  lite::Tensor strides_tensor(kNumberTypeInt32, {3});
  int strides_data[] = {1, 2, 1};
  strides_tensor.set_data(strides_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor, &begins_tensor, &ends_tensor, &strides_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  InitStridedSliceParam(parameter, &in_tensor, &begins_tensor, &ends_tensor, &strides_tensor);
  parameter->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  int8_t expect[4] = {-6, -5, 7, 8};
  for (unsigned int i = 0; i < sizeof(expect); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  in_tensor.set_data(nullptr);
  begins_tensor.set_data(nullptr);
  ends_tensor.set_data(nullptr);
  strides_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
