/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "view/selectview_strides_calc.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace ops {
class TestViewSelectView : public TestView {
 public:
  TestViewSelectView() {}
};

/// Feature: SelectView strides calculator
/// Description: Test view SelectView strides calculator is right
/// Expectation: success
TEST_F(TestViewSelectView, func) {
  auto prim = std::make_shared<Primitive>("SelectView");

  std::vector<int64_t> tensor_data(8, 1);
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({2, 4});
  auto index = std::make_shared<Int64Imm>(1);
  auto dim = std::make_shared<Int64Imm>(0);
  auto storage_info_list = SelectViewCalc(prim, {input_tensor, index, dim});
  ASSERT_EQ(storage_info_list.size(), 1);

  auto dim_tensor = tensor::from_scalar(0, kInt64);
  auto another_storage_info_list = SelectViewCalc(prim, {input_tensor, index, dim_tensor});
  ASSERT_EQ(another_storage_info_list.size(), 1);
  storage_info_list.push_back(another_storage_info_list[0]);

  std::vector<int64_t> expect_shape{4};
  std::vector<int64_t> expect_strides{1};
  for (size_t i = 0; i < storage_info_list.size(); ++i) {
    ASSERT_TRUE(storage_info_list[i]->is_contiguous);
    ASSERT_TRUE(storage_info_list[i]->shape == expect_shape);
    ASSERT_TRUE(storage_info_list[i]->strides == expect_strides);
    ASSERT_TRUE(storage_info_list[i]->storage_offset == 4);
  }
}
}  // namespace ops
}  // namespace mindspore