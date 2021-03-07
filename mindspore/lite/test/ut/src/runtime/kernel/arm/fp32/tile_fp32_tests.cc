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
#include "mindspore/lite/nnacl/base/tile_base.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestTileFp32 : public mindspore::CommonTest {
 public:
  TestTileFp32() {}
};

TEST_F(TestTileFp32, Tile) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {2, 2});
  lite::Tensor out_tensor(kNumberTypeFloat32, {4, 6});
  float input_data[] = {1, 2, 3, 4};
  float output_data[24] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  TileParameter parameter = {0};
  parameter.in_dim_ = 2;
  parameter.in_shape_[0] = 2;
  parameter.in_shape_[1] = 2;
  parameter.multiples_[0] = 2;
  parameter.multiples_[1] = 3;
  parameter.in_strides_[0] = 2;
  parameter.in_strides_[1] = 1;
  parameter.out_strides_[0] = 6;
  parameter.out_strides_[1] = 1;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_TileFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4};
  for (int i = 0; i < 24; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestTileFp32, SimpleTile1) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {2, 2});
  lite::Tensor out_tensor(kNumberTypeFloat32, {4, 2});
  float input_data[] = {1, 2, 3, 4};
  float output_data[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  TileParameter parameter = {0};
  parameter.in_dim_ = 2;
  parameter.in_shape_[0] = 2;
  parameter.in_shape_[1] = 2;
  parameter.multiples_[0] = 2;
  parameter.multiples_[1] = 1;
  parameter.in_strides_[0] = 2;
  parameter.in_strides_[1] = 1;
  parameter.out_strides_[0] = 2;
  parameter.out_strides_[1] = 1;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_TileFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto context = ctx.get();
  context->thread_num_ = 2;
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), context, desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {1, 2, 3, 4, 1, 2, 3, 4};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestTileFp32, SimpleTile2) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {2, 2});
  lite::Tensor out_tensor(kNumberTypeFloat32, {2, 4});
  float input_data[] = {1, 2, 3, 4};
  float output_data[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  TileParameter parameter = {0};
  parameter.in_dim_ = 2;
  parameter.in_shape_[0] = 2;
  parameter.in_shape_[1] = 2;
  parameter.multiples_[0] = 1;
  parameter.multiples_[1] = 2;
  parameter.in_strides_[0] = 2;
  parameter.in_strides_[1] = 1;
  parameter.out_strides_[0] = 4;
  parameter.out_strides_[1] = 1;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_TileFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto context = ctx.get();
  context->thread_num_ = 2;
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), context, desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {1, 2, 1, 2, 3, 4, 3, 4};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
