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
#include "ir/tensor_new.h"
#include "view/reshape_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewReshape : public TestView {
 public:
  TestViewReshape() {}
};

/// Feature: Reshape strides calculator
/// Description: Test view Reshape strides calculator is right
/// Expectation: success
TEST_F(TestViewReshape, ReshapeFunc) {
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({2, 4});
  std::vector<int64_t> new_shape = {1, 4, 2};
  auto storage_info = ReshapeBasicTypeCalc(input_tensor, new_shape);
  std::vector<int64_t> expect_shape({1, 4, 2});
  std::vector<int64_t> expect_strides({8, 2, 1});
  size_t expect_offset = 0;
  ASSERT_TRUE(storage_info != nullptr);
  ASSERT_TRUE(storage_info->is_contiguous);
  ASSERT_TRUE(storage_info->shape == expect_shape);
  ASSERT_TRUE(storage_info->strides == expect_strides);
  ASSERT_TRUE(storage_info->storage_offset == expect_offset);

  // twice dim to infer
  ASSERT_THROW(ReshapeBasicTypeCalc(input_tensor, {-1, -1, 4}), std::exception);
  // invalid dim
  ASSERT_THROW(ReshapeBasicTypeCalc(input_tensor, {-1, -3, 4}), std::exception);
  // num of new shape is greater than origin
  ASSERT_THROW(ReshapeBasicTypeCalc(input_tensor, {2, 2, 4}), std::exception);

  // infer -1 for empty tensor
  std::vector<int64_t> empty_data{};
  auto empty_tensor = tensor::from_vector(empty_data, kInt64);
  empty_tensor->set_shape({0, 4, 2});
  storage_info = ReshapeBasicTypeCalc(empty_tensor, {-1, 0, 4});
  ASSERT_TRUE(storage_info != nullptr);
  std::vector<int64_t> infered_shape{0, 0, 4};
  ASSERT_TRUE(storage_info->shape == infered_shape);
}
}  // namespace ops
}  // namespace mindspore
