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
#include "view/transpose_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewTranspose : public TestView {
 public:
  TestViewTranspose() {}
};

/// Feature: transpose strides calculator
/// Description: Test view transpose strides calculator is right
/// Expectation: success
TEST_F(TestViewTranspose, TransposeFunc) {
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({2, 3});
  std::vector<int64_t> input_perm({1, 0});
  auto storage_info = TransposeBasicTypeCalc(input_tensor, input_perm);
  std::vector<int64_t> expect_out({3, 2});
  ASSERT_FALSE(storage_info.empty());
  ASSERT_FALSE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_out);
}
}  // namespace ops
}  // namespace mindspore
