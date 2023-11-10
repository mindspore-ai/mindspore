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
#include "mindspore/core/ops/view/split_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewSplit : public TestView {
 public:
  TestViewSplit() {}
};

/// Feature: Split strides calculator
/// Description: Test view Split strides calculator is right
/// Expectation: success
TEST_F(TestViewSplit, SplitFunction) {
  auto prim = std::make_shared<Primitive>("Split");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({1, 2, 3});

  int64_t perm_1 = 2;
  int64_t axis_1 = 1;
  auto input_perm = MakeValue(perm_1);
  auto input_axis = MakeValue(axis_1);
  auto storage_list = SplitCalc(prim, std::vector<ValuePtr>({input_tensor, input_axis, input_perm}));
  std::vector<int64_t> expect_shape({1, 1, 3});
  std::vector<int64_t> expect_strides({6, 3, 1});
  size_t expect_offset_1 = 0;
  size_t expect_offset_2 = 3;
  size_t expect_size = 2;
  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides);
  ASSERT_TRUE(storage_list[0]->storage_offset == expect_offset_1);
  ASSERT_FALSE(storage_list[1]->is_contiguous);
  ASSERT_TRUE(storage_list[1]->shape == expect_shape);
  ASSERT_TRUE(storage_list[1]->strides == expect_strides);
  ASSERT_TRUE(storage_list[1]->storage_offset == expect_offset_2);

  int64_t perm_2 = 1;
  int64_t axis_2 = 0;
  input_perm = MakeValue(perm_2);
  input_axis = MakeValue(axis_2);
  storage_list = SplitCalc(prim, std::vector<ValuePtr>({input_tensor, input_axis, input_perm}));
  std::vector<int64_t> expect_shape_2({1, 2, 3});
  std::vector<int64_t> expect_strides_2({6, 3, 1});
  size_t expect_size_2 = 1;
  ASSERT_EQ(storage_list.size(), expect_size_2);
  ASSERT_TRUE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape_2);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides_2);
  ASSERT_TRUE(storage_list[0]->storage_offset == expect_offset_1);
}
}  // namespace ops
}  // namespace mindspore
