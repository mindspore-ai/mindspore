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
#include "mindspore/core/ops/view/squeeze_strides_calc.h"

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
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({2, 1, 1, 5});

  std::vector<int64_t> axis_data = {2};
  auto input_axis = MakeValue(axis_data);

  // test size
  std::vector<ValuePtr> inputs_squeeze_size;
  inputs_squeeze_size.push_back(input_tensor);
  inputs_squeeze_size.push_back(input_axis);

  ASSERT_TRUE(SqueezeCalc(prim, inputs_squeeze_size).empty());

  // test nullptr
  std::vector<ValuePtr> inputs_squeeze_null;
  auto nullinput_tensor = nullptr;
  inputs_squeeze_null.push_back(nullinput_tensor);
  ASSERT_TRUE(SqueezeCalc(prim, inputs_squeeze_null).empty());

  // test type
  std::vector<ValuePtr> inputs_squeeze_type;
  inputs_squeeze_type.push_back(input_axis);
  ASSERT_TRUE(SqueezeCalc(prim, inputs_squeeze_type).empty());

  std::vector<ValuePtr> inputs_squeeze;
  inputs_squeeze.push_back(input_tensor);
  prim->AddAttr(kAxis, input_axis);
  auto storage_info = SqueezeCalc(prim, inputs_squeeze);
  std::vector<int64_t> expect_shape({2, 1, 5});

  ASSERT_FALSE(storage_info.empty());
  ASSERT_TRUE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_shape);

  axis_data = {};
  input_axis = MakeValue(axis_data);
  prim->AddAttr(kAxis, input_axis);
  storage_info = SqueezeCalc(prim, inputs_squeeze);
  std::vector<int64_t> expect_shape_2({2, 5});
  ASSERT_TRUE(storage_info[0]->shape == expect_shape_2);
}

}  // namespace ops
}  // namespace mindspore
