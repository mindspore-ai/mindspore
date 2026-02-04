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
#include "view/diagonal_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewDiagonal : public TestView {
 public:
  TestViewDiagonal() {}
};

/// Feature: Diagonal strides calculator
/// Description: Test view Diagonal strides calculator is right
/// Expectation: success
TEST_F(TestViewDiagonal, View) {
  auto prim = std::make_shared<Primitive>("Diagonal");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({1, 2, 4});
  int64_t input_offset = 1;
  int64_t input_dim1 = 1;
  int64_t input_dim2 = 2;
  auto offset_ = MakeValue(input_offset);
  auto dim1_ = MakeValue(input_dim1);
  auto dim2_ = MakeValue(input_dim2);
  std::vector<ValuePtr> inputs_a;
  inputs_a.emplace_back(input_tensor);
  inputs_a.emplace_back(offset_);
  inputs_a.emplace_back(dim1_);
  inputs_a.emplace_back(dim2_);
  auto storage_info = DiagonalCalc(prim, inputs_a);
  std::vector<int64_t> expect_shape({1, 2});
  std::vector<int64_t> expect_strides({8, 5});
  size_t expect_offset = 1;
  ASSERT_FALSE(storage_info.empty());
  ASSERT_FALSE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_shape);
  ASSERT_TRUE(storage_info[0]->strides == expect_strides);
  ASSERT_TRUE(storage_info[0]->storage_offset == expect_offset);

  // dim1 == dim2
  dim2_ = dim1_;
  ASSERT_THROW(DiagonalCalc(prim, {input_tensor, offset_, dim1_, dim2_}), std::exception);

  // input's rank < 2
  dim2_ = MakeValue(input_dim2);
  input_tensor->set_shape({8});
  ASSERT_THROW(DiagonalCalc(prim, {input_tensor, offset_, dim1_, dim2_}), std::exception);
}
}  // namespace ops
}  // namespace mindspore
