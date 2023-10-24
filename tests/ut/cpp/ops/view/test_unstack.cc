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
#include "mindspore/core/ops/view/unstack_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewUnstack : public TestView {
 public:
  TestViewUnstack() {}
};

/// Feature: unstack strides calculator
/// Description: Test view unstack strides calculator is right
/// Expectation: success
TEST_F(TestViewUnstack, View) {
  auto prim = std::make_shared<Primitive>("Unstack");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({2, 1, 5});

  int64_t axis_data = 0;
  auto input_axis = MakeValue(axis_data);

  // test size
  std::vector<ValuePtr> inputs_unstack_size;
  inputs_unstack_size.push_back(input_tensor);
  inputs_unstack_size.push_back(input_axis);
  ASSERT_TRUE(UnstackCalc(prim, inputs_unstack_size).empty());

   // test nullptr
  std::vector<ValuePtr> inputs_unstack_null;
  auto nullinput_tensor = nullptr;
  inputs_unstack_null.push_back(nullinput_tensor);
  ASSERT_TRUE(UnstackCalc(prim, inputs_unstack_null).empty());

  // test type
  std::vector<ValuePtr> inputs_unstack_type;
  inputs_unstack_type.push_back(input_axis);
  ASSERT_TRUE(UnstackCalc(prim, inputs_unstack_type).empty());

  std::vector<ValuePtr> inputs_unstack;
  inputs_unstack.push_back(input_tensor);
  prim->AddAttr(kAxis, input_axis);
  auto storage_info_vec = UnstackCalc(prim, inputs_unstack);
  std::vector<int64_t> expect_shape({1, 5});

  ASSERT_TRUE(storage_info_vec.size() == 2);

  for (TensorStorageInfoPtr storage_info : storage_info_vec) {
    ASSERT_TRUE(storage_info->is_contiguous);
    ASSERT_TRUE(storage_info->shape == expect_shape);
  }

  // test is_contiguous
  axis_data = 2;
  input_axis = MakeValue(axis_data);
  prim->AddAttr(kAxis, input_axis);
  storage_info_vec = UnstackCalc(prim, inputs_unstack);
  std::vector<int64_t> expect_shape_2({2, 1});
  ASSERT_TRUE(storage_info_vec.size() == 5);
  for (TensorStorageInfoPtr storage_info : storage_info_vec) {
    ASSERT_FALSE(storage_info->is_contiguous);
    ASSERT_TRUE(storage_info->shape == expect_shape_2);
  }
}

}  // namespace ops
}  // namespace mindspore
