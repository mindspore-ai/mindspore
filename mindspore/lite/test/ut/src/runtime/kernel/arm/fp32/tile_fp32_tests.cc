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
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/tile.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestTileFp32 : public mindspore::CommonTest {
 public:
  TestTileFp32() {}
};

TEST_F(TestTileFp32, Tile) {
  lite::tensor::Tensor in_tensor(kNumberTypeFloat32, {2, 2});
  lite::tensor::Tensor out_tensor(kNumberTypeFloat32, {4, 6});
  float input_data[] = {1, 2, 3, 4};
  float output_data[24] = {0};
  in_tensor.SetData(input_data);
  out_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

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

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Tile};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::Context>();
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4};
  for (int i = 0; i < 24; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
}
}  // namespace mindspore
