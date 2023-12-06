/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "test_view.h"
#include "mindspore/core/ops/view/reshape_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewReshape : public TestView {
 public:
  TestViewReshape() {}
};

/// Feature: Reshape strides calculator
/// Description: Test view Reshape strides calculator is right
/// Expectation: success
TEST_F(TestViewReshape, View) {
  auto prim = std::make_shared<Primitive>("Reshape");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({2, 4});
  std::vector<int64_t> input_perm_data = {1, 4, 2};
  auto input_perm = MakeValue(input_perm_data);
  std::vector<ValuePtr> inputs_a;
  inputs_a.emplace_back(input_tensor);
  inputs_a.emplace_back(input_perm);
  auto storage_info = ReshapeCalc(prim, inputs_a);
  std::vector<int64_t> expect_shape({1, 4, 2});
  std::vector<int64_t> expect_strides({8, 2, 1});
  size_t expect_offset = 0;
  ASSERT_FALSE(storage_info.empty());
  ASSERT_TRUE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_shape);
  ASSERT_TRUE(storage_info[0]->strides == expect_strides);
  ASSERT_TRUE(storage_info[0]->storage_offset == expect_offset);
}
}  // namespace ops
}  // namespace mindspore
