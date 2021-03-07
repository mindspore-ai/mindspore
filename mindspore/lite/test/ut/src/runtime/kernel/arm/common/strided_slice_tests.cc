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

#include <memory>
#include "common/common_test.h"
#include "nnacl/fp32/strided_slice_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestStridedSlice : public mindspore::CommonTest {
 public:
  TestStridedSlice() {}
};

void InitStridedSliceParam(StridedSliceParameter *strided_slice_param) {
  strided_slice_param->begins_[0] = 0;
  strided_slice_param->begins_[1] = 0;
  strided_slice_param->begins_[2] = 0;

  strided_slice_param->ends_[0] = 1;
  strided_slice_param->ends_[1] = 2;
  strided_slice_param->ends_[2] = 4;

  strided_slice_param->strides_[0] = 1;
  strided_slice_param->strides_[1] = 2;
  strided_slice_param->strides_[2] = 2;

  strided_slice_param->in_shape_[0] = 1;
  strided_slice_param->in_shape_[1] = 2;
  strided_slice_param->in_shape_[2] = 4;
  strided_slice_param->num_axes_ = 3;
}

TEST_F(TestStridedSlice, StridedSlice) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {1, 2, 4});
  lite::Tensor out_tensor(kNumberTypeFloat32, {1, 1, 2});
  float input_data[] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};
  float output_data[2] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  StridedSliceParameter parameter = {0};
  InitStridedSliceParam(&parameter);
  parameter.op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[2] = {0.2390374, 0.05051243};
  ASSERT_EQ(0, CompareOutputData(output_data, expect, 2, 0.000001));

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestStridedSlice, StridedSliceInt8) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 3, 4});
  lite::Tensor out_tensor(kNumberTypeInt8, {2, 3, 4});
  int8_t input_data[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int8_t output_data[4] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  StridedSliceParameter parameter = {0};
  parameter.begins_[0] = 0;
  parameter.begins_[1] = 1;
  parameter.begins_[2] = 2;
  parameter.ends_[0] = 2;
  parameter.ends_[1] = 3;
  parameter.ends_[2] = 4;
  parameter.strides_[0] = 1;
  parameter.strides_[1] = 2;
  parameter.strides_[2] = 1;
  parameter.in_shape_[0] = 2;
  parameter.in_shape_[1] = 3;
  parameter.in_shape_[2] = 4;
  parameter.num_axes_ = 3;

  parameter.op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_StridedSlice};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect[4] = {-6, -5, 7, 8};
  for (unsigned int i = 0; i < sizeof(expect); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
