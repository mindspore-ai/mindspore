/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/map_tensor.h"

namespace mindspore {
using tensor::Tensor;
using tensor::TensorPtr;

class TestMapTensor : public UT::Common {
 public:
  TestMapTensor() = default;
  ~TestMapTensor() = default;
};

/// Feature: MapTensor
/// Description: test MapTensor API.
/// Expectation: MapTensor API work as expected.
TEST_F(TestMapTensor, TestApi) {
  auto default_value = std::make_shared<StringImm>("zeros");
  auto m = std::make_shared<MapTensor>(kNumberTypeInt32, kNumberTypeFloat32, ShapeVector{4}, default_value);
  ASSERT_TRUE(m != nullptr);
  ASSERT_EQ(m->key_dtype(), kNumberTypeInt32);
  ASSERT_EQ(m->value_dtype(), kNumberTypeFloat32);
  ASSERT_EQ(m->value_shape(), ShapeVector{4});
  ASSERT_EQ(m->default_value(), default_value);
}
}  // namespace mindspore
