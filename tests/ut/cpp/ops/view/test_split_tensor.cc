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
#include "view/split_tensor_view_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewSplitTensor : public TestView {
 public:
  TestViewSplitTensor() {}
};

/// Feature: SplitTensor strides calculator
/// Description: Test view SplitTensor strides calculator is right
/// Expectation: success
TEST_F(TestViewSplitTensor, SplitTensorFunction) {
  // input's rank is 0
  auto scalar_tensor = tensor::from_scalar(1, kInt64);
  ASSERT_THROW(SplitTensorViewBasicTypeCalc(scalar_tensor, 1, 1), std::exception);
  // split_size <= 0
  std::vector<int64_t> tensor_data{1, 2, 3, 4};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({1, 4});
  ASSERT_THROW(SplitTensorViewBasicTypeCalc(input_tensor, -1, 1), std::exception);
}
}  // namespace ops
}  // namespace mindspore