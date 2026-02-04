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
#include "view/copy_strides_calc.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace ops {
class TestViewCopyWithSlice : public TestView {
 public:
  TestViewCopyWithSlice() {}
};

/// Feature: CopyWithSlice strides calculator
/// Description: Test view CopyWithSlice strides calculator
/// Expectation: success
TEST_F(TestViewCopyWithSlice, CopyWithSliceFunc) {
  auto prim = std::make_shared<Primitive>("CopyWithSlice");

  std::vector<int64_t> input_data = {1, 2, 3, 4};
  auto self_tensor = tensor::from_vector(input_data, kInt64);
  self_tensor->set_shape({1, 4});

  std::vector<int64_t> other_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto src_tensor = tensor::from_vector(other_data, kInt64);
  src_tensor->set_shape({2, 1, 4});

  // self.shape != src.shape
  ASSERT_THROW(CopyWithSliceCalc(prim, std::vector<ValuePtr>({self_tensor, src_tensor})), std::exception);

  // self.dtype != src.dtype
  src_tensor = tensor::from_vector(input_data, kFloat32);
  src_tensor->set_shape({1, 4});
  ASSERT_THROW(CopyWithSliceCalc(prim, std::vector<ValuePtr>({self_tensor, src_tensor})), std::exception);
}
}  // namespace ops
}  // namespace mindspore