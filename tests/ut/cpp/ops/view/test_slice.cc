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
#include "mindspore/core/ops/view/slice_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewSlice : public TestView {
 public:
  TestViewSlice() {}
};

/// Feature: Slice strides calculator
/// Description: Test view Slice strides calculator is right
/// Expectation: success
TEST_F(TestViewSlice, SliceFunction) {
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({1, 2, 3});

  auto beigin_pos = std::vector<int64_t>({0, 1, 2});
  auto slice_size = std::vector<int64_t>({1, 1, 1});
  auto begin_pos_perm = MakeValue(beigin_pos);
  auto slice_size_perm = MakeValue(slice_size);
  auto storage_list = SliceCalc(nullptr, std::vector<ValuePtr>({input_tensor, begin_pos_perm, slice_size_perm}));
  std::vector<int64_t> expect_shape_1({1, 1, 1});
  std::vector<int64_t> expect_strides_1({6, 3, 1});
  size_t expect_offset = 5;
  size_t expect_size = 1;
  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_TRUE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape_1);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides_1);
  ASSERT_TRUE(storage_list[0]->storage_offset == expect_offset);

  auto beigin_pos_2 = std::vector<int64_t>({0, 1, 2});
  auto slice_size_2 = std::vector<int64_t>({1, 0, 1});
  begin_pos_perm = MakeValue(beigin_pos_2);
  slice_size_perm = MakeValue(slice_size_2);
  storage_list = SliceCalc(nullptr, std::vector<ValuePtr>({input_tensor, begin_pos_perm, slice_size_perm}));
  std::vector<int64_t> expect_shape_2({1, 0, 1});
  std::vector<int64_t> expect_strides_2({6, 3, 1});
  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape_2);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides_2);
  ASSERT_TRUE(storage_list[0]->storage_offset == expect_offset);
}
}  // namespace ops
}  // namespace mindspore
