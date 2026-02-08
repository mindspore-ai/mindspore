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
#include "view/squeeze_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewSqueeze : public TestView {
 public:
  TestViewSqueeze() {}
};

/// Feature: squeeze strides calculator
/// Description: Test view squeeze strides calculator is right
/// Expectation: success
TEST_F(TestViewSqueeze, View) {
  auto prim = std::make_shared<Primitive>("Squeeze");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({2, 1, 1, 5});

  std::vector<int64_t> axis = {2};

  // test nullptr
  ASSERT_THROW(SqueezeBasicTypeCalc(nullptr, axis), std::exception);

  auto storage_info = SqueezeBasicTypeCalc(input_tensor, axis);
  std::vector<int64_t> expect_shape({2, 1, 5});
  ASSERT_FALSE(storage_info.empty());
  ASSERT_TRUE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_shape);

  storage_info = SqueezeBasicTypeCalc(input_tensor, {});
  std::vector<int64_t> expect_shape_2({2, 5});
  ASSERT_TRUE(storage_info[0]->shape == expect_shape_2);

  input_tensor = tensor::from_scalar(2, kFloat32);
  storage_info = SqueezeBasicTypeCalc(input_tensor, {});
  std::vector<int64_t> expect_shape_3({});
  ASSERT_TRUE(storage_info[0]->shape == expect_shape_3);
}

}  // namespace ops
}  // namespace mindspore
