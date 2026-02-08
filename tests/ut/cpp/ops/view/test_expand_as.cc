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
#include "view/expand_as_strides_calc.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace ops {
class TestViewExpandAs : public TestView {
 public:
  TestViewExpandAs() {}
};

/// Feature: ExpandAs strides calculator
/// Description: Test view ExpandAs strides calculator
/// Expectation: success
TEST_F(TestViewExpandAs, func) {
  std::vector<int64_t> input_data = {1, 2, 3, 4};
  auto input_tensor = tensor::from_vector(input_data, kInt64);
  input_tensor->set_shape({1, 4});

  std::vector<int64_t> other_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto other_tensor = tensor::from_vector(other_data, kInt64);
  other_tensor->set_shape({2, 1, 4});

  auto storage_list = ExpandAsBasicTypeCalc(input_tensor, other_tensor);
  std::vector<int64_t> expect_shape({2, 1, 4});
  std::vector<int64_t> expect_strides({0, 4, 1});
  size_t expect_size = 1;

  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides);
}
}  // namespace ops
}  // namespace mindspore
